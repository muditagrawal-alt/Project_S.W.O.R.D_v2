"""
Interval-label review tool (backend). Pages: /labeler. Labels follow fusion/labels.py; workflow and level definitions
are in fusion/LABELING_GUIDELINES.md.

Derived labels (research/data/labels/*_derived.csv) are shown as pre-labels; whatever an annotator saves is stored
in the database (table label_intervals, with annotator and time, never overwritten — each save is a new revision)
and recorded in the audit log. GET /api/label/export returns the latest revision of every reviewed video as CSV.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func, select
from sqlalchemy.orm import Mapped, mapped_column

from fusion.labels import FIELDS, Interval, load_labels
from fusion.rules import LEVELS
from service.db import Base, audit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"


class LabelInterval(Base):
    __tablename__ = "label_intervals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video: Mapped[str] = mapped_column(String(300), index=True)
    revision: Mapped[int] = mapped_column(Integer, index=True)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(32))
    scope: Mapped[str] = mapped_column(String(10), default="scene")
    anchor: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(120))
    notes: Mapped[str] = mapped_column(Text, default="")
    annotator: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class IntervalIn(BaseModel):
    start_s: float = Field(..., ge=0)
    end_s: float = Field(..., ge=0)
    label: str
    scope: str = "scene"
    anchor: Optional[dict] = None
    verified_from_derived: bool = False   # True: a derived interval the annotator checked (kept or corrected)
    notes: str = ""


class SaveRequest(BaseModel):
    video: str
    annotator: str = Field(..., min_length=1, max_length=80)
    intervals: List[IntervalIn]


def _media_path(video_id: str, data_root: Path) -> Optional[Path]:
    """Map a label video id to a local file under research/data (no path escapes)."""
    parts = video_id.split("/")
    if parts[0] == "mendeley":
        p = data_root / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset" / Path(*parts[1:])
    elif parts[0] == "ucf":
        p = data_root / "ucf_crime" / "extracted" / "UCF_Crimes" / "Videos" / Path(*parts[1:])
    else:
        return None
    p = p.resolve()
    return p if p.is_file() and p.is_relative_to(data_root.resolve()) else None


def register(app: FastAPI, S, data_root: Path = PROJECT_ROOT / "research" / "data") -> None:
    labels_dir = data_root / "labels"

    def derived() -> dict:
        out = {}
        for f in sorted(labels_dir.glob("*_derived.csv")):   # *_TEST_ONLY files are not offered for review
            for iv in load_labels(f):
                out.setdefault(iv.video, []).append(iv)
        return out

    def latest(s, video: str):
        rev = s.scalar(select(func.max(LabelInterval.revision)).where(LabelInterval.video == video))
        if rev is None:
            return None, []
        rows = s.scalars(select(LabelInterval).where(LabelInterval.video == video, LabelInterval.revision == rev)
                         .order_by(LabelInterval.start_s)).all()
        return rev, rows

    @app.get("/labeler", response_class=HTMLResponse)
    def labeler_page():
        return (STATIC / "labeler.html").read_text()

    @app.get("/api/label/videos")
    def label_videos(dataset: Optional[str] = None):
        d = derived()
        with S() as s:
            reviewed = dict(s.execute(select(LabelInterval.video, func.max(LabelInterval.revision)).group_by(LabelInterval.video)).all())
        vids = []
        for v in sorted(set(d) | set(reviewed)):
            if dataset and not v.startswith(dataset + "/"):
                continue
            vids.append({"video": v, "derived_intervals": len(d.get(v, [])), "reviewed_revision": reviewed.get(v),
                         "media": _media_path(v, data_root) is not None,
                         "needs_review": any("NEEDS REVIEW" in iv.notes for iv in d.get(v, []))})
        return vids

    @app.get("/api/label/intervals")
    def label_intervals(video: str):
        with S() as s:
            rev, rows = latest(s, video)
            if rows:
                return {"video": video, "revision": rev, "origin": "reviewed",
                        "intervals": [{"start_s": r.start_s, "end_s": r.end_s, "label": r.label, "scope": r.scope,
                                       "anchor": r.anchor, "source": r.source, "notes": r.notes, "annotator": r.annotator}
                                      for r in rows]}
        d = derived().get(video, [])
        return {"video": video, "revision": None, "origin": "derived",
                "intervals": [{"start_s": iv.start_s, "end_s": iv.end_s, "label": iv.label, "scope": iv.scope,
                               "anchor": iv.anchor, "source": iv.source, "notes": iv.notes, "annotator": None} for iv in d]}

    @app.get("/api/label/media")
    def label_media(video: str):
        p = _media_path(video, data_root)
        if p is None:
            raise HTTPException(404, "media not available locally")
        return FileResponse(p, media_type="video/mp4")

    @app.put("/api/label/intervals")
    def save_intervals(req: SaveRequest):
        if not req.intervals:
            raise HTTPException(422, "at least one interval (use NORMAL for a reviewed video with nothing to flag)")
        for iv in req.intervals:   # validates level, scope, ordering, anchors
            try:
                Interval(req.video, iv.start_s, iv.end_s, iv.label, iv.scope, iv.anchor)
            except ValueError as e:
                raise HTTPException(422, str(e))
        with S() as s:
            rev = (s.scalar(select(func.max(LabelInterval.revision)).where(LabelInterval.video == req.video)) or 0) + 1
            for iv in req.intervals:
                src = f"human-verified:{req.annotator}" if iv.verified_from_derived else f"human:{req.annotator}"
                s.add(LabelInterval(video=req.video, revision=rev, start_s=iv.start_s, end_s=iv.end_s, label=iv.label,
                                    scope=iv.scope, anchor=iv.anchor, source=src, notes=iv.notes, annotator=req.annotator))
            audit(s, req.annotator, "labels_saved", f"video:{req.video}", revision=rev, intervals=len(req.intervals))
            s.commit()
        return {"video": req.video, "revision": rev, "saved": len(req.intervals)}

    @app.get("/api/label/export", response_class=PlainTextResponse)
    def export_labels(annotator: Optional[str] = None):
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=FIELDS)
        wr.writeheader()
        with S() as s:
            videos = [v for (v,) in s.execute(select(LabelInterval.video).distinct())]
            for v in sorted(videos):
                _, rows = latest(s, v)
                for r in rows:
                    if annotator and r.annotator != annotator:
                        continue
                    wr.writerow({"video": r.video, "start_s": round(r.start_s, 3), "end_s": round(r.end_s, 3), "label": r.label,
                                 "scope": r.scope, "anchor": json.dumps(r.anchor) if r.anchor else "", "source": r.source,
                                 "notes": r.notes})
        return buf.getvalue()

    @app.get("/api/label/levels")
    def label_levels():
        return LEVELS

"""
Interval labels for training and evaluating Phase 5 (rules baseline and Temporal Transformer).

A label says what is observable in a video during [start_s, end_s] — never who someone is or what they intend:

    NORMAL                no threat-relevant behaviour
    ELEVATED              unusual but not threatening (running, loitering in a restricted zone, a weapon-like object
                          not held by anyone)
    SUSPICIOUS_ACTIVITY   physical aggression or threatening behaviour without a visible weapon (fighting, assault)
    WEAPON_RELATED_EVENT  a weapon is visibly carried, drawn, aimed or used

CSV columns (one interval per row; intervals may overlap, the highest level wins):

    video       file name of the video / sequence the times refer to
    start_s     interval start, seconds from the start of the video
    end_s       interval end, seconds
    label       one of the four levels above
    scope       "scene" (default: the whole frame) or "person"
    anchor      for scope=person: JSON {"t": seconds, "bbox": [x1, y1, x2, y2]} locating the person at one instant;
                it is matched to whatever tracker produced the features (track IDs are not stable across trackers)
    source      provenance: "derived:<rule>" (automatic, unverified), "human:<annotator>", or "human-verified:<annotator>"
    notes       free text (e.g. dataset action code)

Unlabelled time inside a labelled video counts as NORMAL only if the video is listed with `covers_video=True` in
the dataset manifest (exhaustive annotation); otherwise it is ignored in training and evaluation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from fusion.rules import LEVELS

FIELDS = ["video", "start_s", "end_s", "label", "scope", "anchor", "source", "notes"]


@dataclass
class Interval:
    video: str
    start_s: float
    end_s: float
    label: str
    scope: str = "scene"
    anchor: Optional[dict] = None
    source: str = "derived"
    notes: str = ""

    def __post_init__(self):
        if self.label not in LEVELS:
            raise ValueError(f"unknown label {self.label!r}")
        if self.scope not in ("scene", "person"):
            raise ValueError(f"unknown scope {self.scope!r}")
        if self.end_s < self.start_s:
            raise ValueError(f"end before start: {self}")
        if self.scope == "person" and not (self.anchor and "bbox" in self.anchor and "t" in self.anchor):
            raise ValueError("person-scope labels need an anchor {t, bbox}")

    @property
    def level(self) -> int:
        return LEVELS.index(self.label)


def save_labels(intervals: Iterable[Interval], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=FIELDS)
        wr.writeheader()
        for iv in intervals:
            row = asdict(iv)
            row["anchor"] = json.dumps(iv.anchor) if iv.anchor else ""
            row["start_s"], row["end_s"] = round(iv.start_s, 3), round(iv.end_s, 3)
            wr.writerow(row)


def load_labels(path: Path) -> List[Interval]:
    out = []
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            out.append(Interval(video=row["video"], start_s=float(row["start_s"]), end_s=float(row["end_s"]),
                                label=row["label"], scope=row.get("scope") or "scene",
                                anchor=json.loads(row["anchor"]) if row.get("anchor") else None,
                                source=row.get("source") or "", notes=row.get("notes") or ""))
    return out


def by_video(intervals: Iterable[Interval]) -> Dict[str, List[Interval]]:
    out: Dict[str, List[Interval]] = {}
    for iv in intervals:
        out.setdefault(iv.video, []).append(iv)
    return out


def scene_levels(intervals: Sequence[Interval], times: Sequence[float], exhaustive: bool = True) -> np.ndarray:
    """Per-time scene level (highest overlapping interval; person-scope labels count for the scene too).
    Unlabelled times: 0 (NORMAL) if exhaustive, else -1 (ignore)."""
    t = np.asarray(times, dtype=float)
    lv = np.full(len(t), 0 if exhaustive else -1, dtype=int)
    for iv in intervals:
        m = (t >= iv.start_s - 1e-9) & (t <= iv.end_s + 1e-9)
        lv[m] = np.maximum(lv[m], iv.level)
    return lv


def merge_adjacent(intervals: List[Interval], gap_s: float = 0.0) -> List[Interval]:
    """Merge same-video, same-label, same-scope intervals that touch or are closer than gap_s."""
    out: List[Interval] = []
    for iv in sorted(intervals, key=lambda i: (i.video, i.scope, i.label, i.start_s)):
        last = out[-1] if out else None
        if (last and last.video == iv.video and last.label == iv.label and last.scope == iv.scope
                and last.anchor is None and iv.anchor is None and iv.start_s - last.end_s <= gap_s):
            last.end_s = max(last.end_s, iv.end_s)
        else:
            out.append(Interval(**{**asdict(iv)}))
    return out


@dataclass
class LabelSetManifest:
    """Describes one label file: which videos it covers exhaustively and how the labels were produced."""
    name: str
    videos: List[str]
    covers_video: bool
    sources: List[str]
    description: str
    extra: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

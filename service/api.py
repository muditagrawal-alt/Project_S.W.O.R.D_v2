"""
Phase 6 — S.W.O.R.D. incident intelligence service (FastAPI + WebSocket + SQL database + FFmpeg evidence).

    python -m service                       # http://127.0.0.1:8000  (dashboard at /)
    uvicorn --factory service.api:create_app --host 127.0.0.1 --port 8000

Environment:
    SWORD_DATABASE_URL  default sqlite:///service_data/sword.db; PostgreSQL: postgresql+psycopg://user:pw@host/sword
    SWORD_DATA_DIR      default service_data/ (run outputs, evidence clips, snapshots)
    SWORD_MEDIA_ROOTS   os.pathsep-separated folders that file sources must be inside (default: the project root)

A run processes a video file, camera index or stream URL through Phases 1-5 (pipeline.system.SwordSystem) on one
worker thread (runs queue; the GPU is shared). Fusion level changes become events, scene escalations become
incidents with an H.264 evidence clip (pre/post roll), an annotated snapshot and SHA-256 hashes. Every state change
is written to the audit log and pushed to WebSocket clients (/ws/events).

There is no authentication: the operator name in requests is self-declared. Bind to 127.0.0.1 (the default) and
add authentication before exposing the service on a network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from fusion.rules import LEVELS
from service import labeling  # noqa: F401  (registers the label_intervals table before create_all)
from service.db import AuditLog, Event, Incident, Run, audit, make_session_factory, to_dict
from service.incidents import IncidentManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"


def _sha256(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _plain(o):
    """numpy -> python for JSON."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class EventHub:
    """Fan-out of JSON messages from worker threads to every connected WebSocket."""

    def __init__(self, history: int = 200):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: set = set()
        self.recent: deque = deque(maxlen=history)

    def publish(self, msg: dict) -> None:
        msg = _plain(msg)
        if msg.get("type") != "frame":
            self.recent.append(msg)
        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._fanout, msg)

    def _fanout(self, msg: dict) -> None:
        for q in list(self.clients):
            if q.qsize() < 1000:   # slow client: drop rather than grow without bound
                q.put_nowait(msg)


class RunRequest(BaseModel):
    source: str = Field(..., description="video file path, camera index (e.g. '0') or stream URL (rtsp://...)")
    actor: str = "operator"
    fps: Optional[float] = Field(None, description="override the source fps (e.g. 2 for 2-fps image sequences)")
    max_frames: Optional[int] = None
    live: Optional[bool] = Field(None, description="wall-clock timestamps + async Phase 4; default: True for cameras/streams")
    use_pose: bool = True
    use_action: bool = True
    use_ensemble: bool = False
    weapon_model: Optional[str] = Field(None, description="a .pt file inside weights/")
    weapon_imgsz: int = Field(640, ge=320, le=2560, description="weapon detector input size")
    save_video: bool = False
    open_level: str = "SUSPICIOUS_ACTIVITY"
    pre_s: float = 5.0
    post_s: float = 3.0


class AckRequest(BaseModel):
    actor: str = "operator"
    note: Optional[str] = None


def _is_stream(source: str) -> bool:
    return source.isdigit() or "://" in source


def default_make_system(fps: float, req: RunRequest, live: bool):
    from pipeline.system import SwordSystem

    kw: Dict[str, Any] = dict(use_pose=req.use_pose, use_action=req.use_action, use_ensemble=req.use_ensemble,
                              async_action=live, weapon_imgsz=req.weapon_imgsz)
    if req.weapon_model:
        kw["weapon_model"] = req.weapon_model
    return SwordSystem(fps=fps, **kw)


class RunWorker:
    def __init__(self, sessions, hub: EventHub, data_dir: Path,
                 make_system: Callable[[float, RunRequest, bool], Any] = default_make_system):
        self.S, self.hub, self.data_dir, self.make_system = sessions, hub, data_dir, make_system
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sword-run")
        self.stop_flags: Dict[str, threading.Event] = {}
        self.previews: Dict[str, bytes] = {}
        self.futures: Dict[str, Any] = {}

    def submit(self, run_id: str, req: RunRequest) -> None:
        self.stop_flags[run_id] = threading.Event()
        self.futures[run_id] = self.pool.submit(self._execute, run_id, req)

    def stop(self, run_id: str) -> bool:
        flag = self.stop_flags.get(run_id)
        if flag is None:
            return False
        flag.set()
        return True

    def shutdown(self) -> None:
        for f in self.stop_flags.values():
            f.set()
        self.pool.shutdown(wait=True, cancel_futures=True)

    # ─────────────────────────── one run ───────────────────────────
    def _execute(self, run_id: str, req: RunRequest) -> None:
        from pipeline.run import draw, run_video

        S, hub = self.S, self.hub
        live = _is_stream(req.source) if req.live is None else req.live
        out_dir = self.data_dir / "runs" / run_id
        with S() as s:
            run = s.get(Run, run_id)
            run.status, run.out_dir = "running", str(out_dir)
            s.commit()
        hub.publish({"type": "run_started", "run_id": run_id, "source": req.source, "live": live})

        state: Dict[str, Any] = {"mgr": None, "system": None, "last": None, "last_pub": 0.0, "fps": None}

        def make(fps: float):
            state["fps"] = fps
            system = self.make_system(fps, req, live)
            state["system"] = system
            state["mgr"] = IncidentManager(out_dir / "evidence", fps, open_level=req.open_level, pre_s=req.pre_s,
                                           post_s=req.post_s, annotate=draw, on_open=on_open, on_update=on_update,
                                           on_close=on_close, on_evidence=on_evidence)
            with S() as s:
                s.get(Run, run_id).fps = fps
                s.commit()
            return system

        def on_open(inc: dict) -> int:
            with S() as s:
                row = Incident(run_id=run_id, status="open", level_start=inc["level_start"], level_peak=inc["level_peak"],
                               start_t=inc["start_t"], start_frame=inc["start_frame"], track_ids=inc["track_ids"],
                               evidence=inc["evidence"])
                s.add(row)
                s.flush()
                audit(s, "system", "incident_opened", f"incident:{row.id}", run_id=run_id, level=inc["level_start"],
                      t=inc["start_t"])
                s.commit()
                iid = row.id
            hub.publish({"type": "incident_opened", "run_id": run_id, "incident": dict(inc, id=iid)})
            return iid

        def on_update(inc: dict) -> None:
            with S() as s:
                row = s.get(Incident, inc["id"])
                row.level_peak, row.track_ids, row.evidence = inc["level_peak"], inc["track_ids"], inc["evidence"]
                audit(s, "system", "incident_escalated", f"incident:{inc['id']}", level=inc["level_peak"])
                s.commit()
            hub.publish({"type": "incident_updated", "run_id": run_id, "incident": inc})

        def on_close(inc: dict) -> None:
            with S() as s:
                row = s.get(Incident, inc["id"])
                row.status, row.end_t, row.end_frame = "closed", inc["end_t"], inc["end_frame"]
                row.level_peak, row.track_ids, row.evidence = inc["level_peak"], inc["track_ids"], inc["evidence"]
                audit(s, "system", "incident_closed", f"incident:{inc['id']}", t=inc["end_t"],
                      reason=inc.get("closed_by", "level_dropped"))
                s.commit()
            hub.publish({"type": "incident_closed", "run_id": run_id, "incident": inc})

        def on_evidence(info: dict) -> None:  # evidence thread
            clip_h, snap_h = _sha256(info.get("clip_path")), _sha256(info.get("snapshot_path"))
            with S() as s:
                row = s.get(Incident, info["id"])
                row.clip_path, row.snapshot_path = info.get("clip_path"), info.get("snapshot_path")
                row.clip_sha256, row.snapshot_sha256 = clip_h, snap_h
                audit(s, "system", "evidence_written", f"incident:{info['id']}", clip=info.get("clip_path"),
                      clip_sha256=clip_h, snapshot_sha256=snap_h, clip_seconds=info.get("clip_seconds"),
                      truncated=info.get("truncated"))
                s.commit()
            hub.publish({"type": "evidence_ready", "run_id": run_id, "incident_id": info["id"],
                         "clip_seconds": info.get("clip_seconds")})

        def on_frame(rec: dict, frame: np.ndarray) -> None:
            mgr = state["mgr"]
            mgr.on_frame(rec, frame)
            state["last"] = rec
            iid = mgr.active.id if mgr.active is not None else next(
                (c.id for c in mgr.closing if c.end_frame == rec["frame_idx"]), None)
            if rec["fusion_events"]:
                with S() as s:
                    for e in rec["fusion_events"]:
                        payload = {k: v for k, v in e.items() if k not in ("t", "scope", "track_id", "from", "to", "type")}
                        s.add(Event(run_id=run_id, incident_id=iid, t=e["t"], frame_idx=rec["frame_idx"], scope=e["scope"],
                                    track_id=e["track_id"], from_level=e["from"], to_level=e["to"], payload=_plain(payload)))
                    s.commit()
                for e in rec["fusion_events"]:
                    hub.publish({"type": "level_change", "run_id": run_id, "frame_idx": rec["frame_idx"],
                                 "incident_id": iid, **e})
            now = time.monotonic()
            if now - state["last_pub"] >= 0.2 or rec["fusion_events"]:
                state["last_pub"] = now
                ok, jpg = cv2.imencode(".jpg", draw(frame, rec), [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.previews[run_id] = jpg.tobytes()
                th = rec["threat"]
                hub.publish({"type": "frame", "run_id": run_id, "frame_idx": rec["frame_idx"], "t": rec["timestamp"],
                             "scene_level": th["scene_level"], "incident_id": iid,
                             "persons": [{"track_id": p["track_id"], "level": th["tracks"].get(p["track_id"], {}).get("level"),
                                          "action": (p.get("action") or {}).get("top1_name")} for p in rec["persons"]],
                             "weapons": [{"cls_name": w["cls_name"], "score": w["score"], "person_id": w["person_id"]}
                                         for w in rec["weapons"]]})

        try:
            res = run_video(req.source, out_dir, gt=None, save_video=req.save_video, max_frames=req.max_frames,
                            fps_override=req.fps, on_frame=on_frame, live=live, keep_frames=not live,
                            should_stop=self.stop_flags[run_id].is_set, make_system=make)
            incidents = state["mgr"].flush(state["last"]) if state["mgr"] else []
            if state["system"] is not None and hasattr(state["system"], "close"):
                state["system"].close()
            stopped = self.stop_flags[run_id].is_set()
            summary = {k: res["summary"].get(k) for k in ("frames", "fps", "live", "wall_seconds", "scene_level_share",
                                                           "person_tracks", "action_updates", "fusion_events", "timing_ms",
                                                           "sources")}
            summary["incidents"] = len(incidents)
            (out_dir / "incidents.json").write_text(json.dumps(_plain(incidents), indent=2))
            with S() as s:
                run = s.get(Run, run_id)
                run.status = "stopped" if stopped else "finished"
                run.finished_at, run.frames, run.summary = datetime.now(timezone.utc), summary["frames"], _plain(summary)
                audit(s, "system", "run_" + run.status, f"run:{run_id}", frames=summary["frames"],
                      incidents=len(incidents))
                s.commit()
            hub.publish({"type": "run_finished", "run_id": run_id, "status": "stopped" if stopped else "finished",
                         "summary": summary})
        except Exception as exc:  # noqa: BLE001 — report any failure on the run
            if state["mgr"] is not None:
                try:
                    state["mgr"].flush(state["last"])
                except Exception:
                    pass
            with S() as s:
                run = s.get(Run, run_id)
                run.status, run.error, run.finished_at = "failed", f"{type(exc).__name__}: {exc}", datetime.now(timezone.utc)
                audit(s, "system", "run_failed", f"run:{run_id}", error=run.error)
                s.commit()
            hub.publish({"type": "run_failed", "run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.stop_flags.pop(run_id, None)


def create_app(database_url: Optional[str] = None, data_dir: Optional[str] = None,
               media_roots: Optional[List[str]] = None,
               make_system: Callable[[float, RunRequest, bool], Any] = default_make_system,
               label_data_root: Optional[str] = None) -> FastAPI:
    data = Path(data_dir or os.environ.get("SWORD_DATA_DIR", PROJECT_ROOT / "service_data")).resolve()
    data.mkdir(parents=True, exist_ok=True)
    url = database_url or os.environ.get("SWORD_DATABASE_URL", f"sqlite:///{data / 'sword.db'}")
    roots_env = os.environ.get("SWORD_MEDIA_ROOTS")
    roots = [Path(r).resolve() for r in (media_roots or (roots_env.split(os.pathsep) if roots_env else [PROJECT_ROOT]))]
    S = make_session_factory(url)
    hub = EventHub()
    worker = RunWorker(S, hub, data, make_system)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.loop = asyncio.get_running_loop()
        yield
        worker.shutdown()

    app = FastAPI(title="S.W.O.R.D. incident intelligence", version="0.1", lifespan=lifespan)
    app.state.hub, app.state.worker, app.state.sessions, app.state.data_dir = hub, worker, S, data

    def _get(model, key):
        with S() as s:
            obj = s.get(model, key)
            if obj is None:
                raise HTTPException(404, f"{model.__name__} {key} not found")
            return obj

    labeling.register(app, S, Path(label_data_root) if label_data_root else PROJECT_ROOT / "research" / "data")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (STATIC / "dashboard.html").read_text()

    @app.get("/api/health")
    def health():
        from service.incidents import ffmpeg_exe

        return {"status": "ok", "database": url.split("://")[0], "ffmpeg": ffmpeg_exe() is not None,
                "levels": LEVELS, "data_dir": str(data)}

    @app.post("/api/runs", status_code=201)
    def create_run(req: RunRequest):
        if req.open_level not in LEVELS[1:]:
            raise HTTPException(422, f"open_level must be one of {LEVELS[1:]}")
        if not _is_stream(req.source):
            src = Path(req.source).expanduser().resolve()
            if not src.is_file():
                raise HTTPException(422, f"source file not found: {req.source}")
            if not any(src.is_relative_to(r) for r in roots):
                raise HTTPException(403, "source is outside SWORD_MEDIA_ROOTS")
            req.source = str(src)
        if req.weapon_model:
            wm = Path(req.weapon_model).expanduser()
            wm = (wm if wm.is_absolute() else PROJECT_ROOT / wm).resolve()
            if not (wm.is_file() and wm.is_relative_to((PROJECT_ROOT / "weights").resolve())):
                raise HTTPException(422, "weapon_model must be an existing file inside weights/")
            req.weapon_model = str(wm)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        with S() as s:
            run = Run(id=run_id, source=req.source, status="queued", options=req.model_dump(exclude={"actor"}))
            s.add(run)
            audit(s, req.actor, "run_created", f"run:{run_id}", source=req.source)
            s.commit()
            out = to_dict(run)
        worker.submit(run_id, req)
        return out

    @app.get("/api/runs")
    def list_runs(limit: int = Query(50, le=500)):
        with S() as s:
            return [to_dict(r) for r in s.scalars(select(Run).order_by(Run.created_at.desc()).limit(limit))]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        return to_dict(_get(Run, run_id))

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str, body: AckRequest = AckRequest()):
        _get(Run, run_id)
        if not worker.stop(run_id):
            raise HTTPException(409, "run is not active")
        with S() as s:
            audit(s, body.actor, "run_stop_requested", f"run:{run_id}")
            s.commit()
        return {"run_id": run_id, "stop_requested": True}

    @app.get("/api/runs/{run_id}/preview.jpg")
    def preview(run_id: str):
        jpg = worker.previews.get(run_id)
        if jpg is None:
            raise HTTPException(404, "no preview yet")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/incidents")
    def list_incidents(run_id: Optional[str] = None, status: Optional[str] = None,
                       unacknowledged: bool = False, limit: int = Query(100, le=1000)):
        with S() as s:
            q = select(Incident).order_by(Incident.id.desc()).limit(limit)
            if run_id:
                q = q.where(Incident.run_id == run_id)
            if status:
                q = q.where(Incident.status == status)
            if unacknowledged:
                q = q.where(Incident.ack_by.is_(None))
            return [to_dict(i) for i in s.scalars(q)]

    @app.get("/api/incidents/{incident_id}")
    def get_incident(incident_id: int):
        with S() as s:
            inc = s.get(Incident, incident_id)
            if inc is None:
                raise HTTPException(404, "incident not found")
            events = s.scalars(select(Event).where(Event.incident_id == incident_id).order_by(Event.id))
            return dict(to_dict(inc), events=[to_dict(e) for e in events])

    @app.post("/api/incidents/{incident_id}/ack")
    def acknowledge(incident_id: int, body: AckRequest):
        with S() as s:
            inc = s.get(Incident, incident_id)
            if inc is None:
                raise HTTPException(404, "incident not found")
            inc.ack_by, inc.ack_at, inc.ack_note = body.actor, datetime.now(timezone.utc), body.note
            audit(s, body.actor, "incident_acknowledged", f"incident:{incident_id}", note=body.note)
            s.commit()
            out = to_dict(inc)
        hub.publish({"type": "incident_acknowledged", "incident_id": incident_id, "actor": body.actor})
        return out

    def _evidence(incident_id: int, kind: str, actor: str):
        inc = _get(Incident, incident_id)
        path = inc.clip_path if kind == "clip" else inc.snapshot_path
        if not path or not Path(path).exists():
            raise HTTPException(404, f"{kind} not available (yet)")
        with S() as s:  # evidence access is part of the chain of custody
            audit(s, actor, f"evidence_{kind}_accessed", f"incident:{incident_id}")
            s.commit()
        return FileResponse(path, media_type="video/mp4" if kind == "clip" else "image/jpeg")

    @app.get("/api/incidents/{incident_id}/clip")
    def clip(incident_id: int, actor: str = "operator"):
        return _evidence(incident_id, "clip", actor)

    @app.get("/api/incidents/{incident_id}/snapshot")
    def snapshot(incident_id: int, actor: str = "operator"):
        return _evidence(incident_id, "snapshot", actor)

    @app.get("/api/audit")
    def audit_log(limit: int = Query(200, le=5000), target: Optional[str] = None):
        with S() as s:
            q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
            if target:
                q = q.where(AuditLog.target == target)
            return [to_dict(a) for a in s.scalars(q)]

    @app.websocket("/ws/events")
    async def ws_events(websocket: WebSocket):
        await websocket.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub.clients.add(q)
        try:
            await websocket.send_text(json.dumps({"type": "hello", "recent": list(hub.recent)[-50:]}))
            while True:
                await websocket.send_text(json.dumps(await q.get()))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(q)

    return app

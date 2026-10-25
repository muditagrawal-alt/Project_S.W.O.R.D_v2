"""
Phase 6 incident manager: scene-level transitions -> incidents with evidence clips and snapshots.

An incident opens when the scene level rises to `open_level` (default SUSPICIOUS_ACTIVITY) and closes when it falls
back below it (the fusion engine's hysteresis already prevents flicker). The clip covers `pre_s` seconds before the
opening frame to `post_s` seconds after closing. Frames are held as JPEG in a ring buffer to bound memory, clips are
encoded on a background thread (OpenCV, then FFmpeg -> H.264 so browsers can play them), and every step reports
through callbacks so the service can persist it and push it to dashboards.

The clip keeps the original pixels (evidence); the snapshot is the annotated opening frame (for the operator).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from fusion.rules import LEVELS


def ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def write_clip(jpegs: List[bytes], fps: float, path: Path) -> Path:
    """Decode JPEG frames -> mp4v via OpenCV -> H.264 (yuv420p, faststart) via FFmpeg when available."""
    first = cv2.imdecode(np.frombuffer(jpegs[0], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    raw = path.with_suffix(".mp4v.mp4")
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for b in jpegs:
        wr.write(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
    wr.release()
    exe = ffmpeg_exe()
    if exe is None:
        raw.replace(path)
        return path
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and path.exists():
        raw.unlink(missing_ok=True)
    else:  # keep the playable-in-VLC mp4v rather than nothing
        raw.replace(path)
    return path


class _Open:
    def __init__(self, incident_id, rec, pre: List[bytes], level: int, tracks: List[int], evidence: dict):
        self.id = incident_id
        self.start_t, self.start_frame = rec["timestamp"], rec["frame_idx"]
        self.level_start = self.level_peak = level
        self.tracks = set(tracks)
        self.evidence = evidence
        self.frames: List[bytes] = list(pre)
        self.closed_at: Optional[float] = None
        self.end_frame: Optional[int] = None
        self.truncated = False
        self.snapshot: Optional[Path] = None


class IncidentManager:
    def __init__(self, out_dir: Path, fps: float, open_level: str = "SUSPICIOUS_ACTIVITY", pre_s: float = 5.0,
                 post_s: float = 3.0, max_clip_s: float = 120.0, jpeg_quality: int = 90,
                 annotate: Optional[Callable[[np.ndarray, dict], np.ndarray]] = None,
                 on_open: Optional[Callable[[dict], Optional[int]]] = None,
                 on_update: Optional[Callable[[dict], None]] = None,
                 on_close: Optional[Callable[[dict], None]] = None,
                 on_evidence: Optional[Callable[[dict], None]] = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps, self.open_level = fps, LEVELS.index(open_level)
        self.pre_s, self.post_s, self.max_frames = pre_s, post_s, int(max_clip_s * fps)
        self.quality = jpeg_quality
        self.annotate = annotate
        self.on_open, self.on_update, self.on_close, self.on_evidence = on_open, on_update, on_close, on_evidence
        self.ring: deque = deque(maxlen=max(1, int(round(pre_s * fps))))
        self.active: Optional[_Open] = None
        self.closing: List[_Open] = []          # closed, still collecting post-roll
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence")
        self._jobs: List[Future] = []
        self._next_local_id = 1
        self._lock = threading.Lock()
        self.incidents: List[dict] = []

    @staticmethod
    def _evidence(rec: dict, level: int) -> tuple:
        tracks = [tid for tid, v in rec["threat"]["tracks"].items() if LEVELS.index(v["level"]) >= level]
        ev: Dict[str, float] = {}
        for tid in tracks:
            for k, v in rec["threat"]["tracks"][tid]["evidence"].items():
                ev[k] = max(ev.get(k, 0.0), v)
        return [int(t) for t in tracks], {k: round(v, 3) for k, v in ev.items() if v > 0.01}

    def _summary(self, inc: _Open) -> dict:
        return {"id": inc.id, "start_t": inc.start_t, "start_frame": inc.start_frame, "end_t": inc.closed_at,
                "end_frame": inc.end_frame, "level_start": LEVELS[inc.level_start], "level_peak": LEVELS[inc.level_peak],
                "track_ids": sorted(inc.tracks), "evidence": inc.evidence, "truncated": inc.truncated}

    def on_frame(self, rec: dict, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        jpg = buf.tobytes() if ok else None
        level = LEVELS.index(rec["threat"]["scene_level"])
        t = rec["timestamp"]

        # post-roll for incidents that already closed
        for inc in list(self.closing):
            if jpg is not None:
                inc.frames.append(jpg)
            if t - inc.closed_at >= self.post_s:
                self.closing.remove(inc)
                self._finalize(inc)

        if self.active is None and level >= self.open_level:
            tracks, ev = self._evidence(rec, level)
            inc = _Open(None, rec, list(self.ring), level, tracks, ev)
            summary = self._summary(inc)
            inc.id = (self.on_open(summary) if self.on_open else None) or self._next_local_id
            self._next_local_id += 1
            summary["id"] = inc.id
            snap = self.out_dir / f"incident_{inc.id}_snapshot.jpg"
            cv2.imwrite(str(snap), self.annotate(frame, rec) if self.annotate else frame)
            inc.snapshot = snap
            self.active = inc
        elif self.active is not None:
            inc = self.active
            tracks, ev = self._evidence(rec, max(self.open_level, min(level, inc.level_peak)))
            inc.tracks.update(tracks)
            changed = level > inc.level_peak
            if changed:
                inc.level_peak = level
            for k, v in ev.items():
                if v > inc.evidence.get(k, 0.0):
                    inc.evidence[k] = v
            if changed and self.on_update:
                self.on_update(self._summary(inc))
            if level < self.open_level:
                inc.closed_at, inc.end_frame = t, rec["frame_idx"]
                if jpg is not None:  # the closing frame belongs to the clip too
                    inc.frames.append(jpg)
                self.active = None
                self.closing.append(inc)
                if self.on_close:
                    self.on_close(self._summary(inc))

        if self.active is not None and jpg is not None:
            if len(self.active.frames) < self.max_frames:
                self.active.frames.append(jpg)
            else:
                self.active.truncated = True
        if jpg is not None:
            self.ring.append(jpg)

    def _finalize(self, inc: _Open) -> None:
        frames, inc.frames = inc.frames, []
        summary = self._summary(inc)
        snap = inc.snapshot

        def job():
            path = self.out_dir / f"incident_{inc.id}.mp4"
            info = dict(summary, clip_path=None, snapshot_path=str(snap) if snap else None, clip_frames=len(frames))
            if frames:
                write_clip(frames, self.fps, path)
                info["clip_path"] = str(path)
                info["clip_seconds"] = round(len(frames) / self.fps, 2)
            with self._lock:
                self.incidents.append(info)
            if self.on_evidence:
                self.on_evidence(info)
            return info

        self._jobs.append(self._pool.submit(job))

    def flush(self, last_rec: Optional[dict] = None) -> List[dict]:
        """End of stream: close the active incident, write every pending clip, wait for the encoder."""
        if self.active is not None:
            inc = self.active
            inc.closed_at = last_rec["timestamp"] if last_rec else inc.start_t
            inc.end_frame = last_rec["frame_idx"] if last_rec else inc.start_frame
            self.active = None
            if self.on_close:
                self.on_close(dict(self._summary(inc), closed_by="end_of_stream"))
            self.closing.append(inc)
        for inc in self.closing:
            self._finalize(inc)
        self.closing = []
        for j in self._jobs:
            j.result()
        self._jobs.clear()
        self._pool.shutdown(wait=True)
        return sorted(self.incidents, key=lambda d: d["start_t"])

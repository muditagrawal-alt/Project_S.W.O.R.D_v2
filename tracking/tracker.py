"""
Multi-object tracking with ByteTrack (Ultralytics implementation), usable with ANY detector.

Ultralytics only exposes ByteTrack through `model.track()`, which ties it to a single YOLO model. Phase 2 needs
it for two independent streams — people (from a COCO person detector) and weapons (from the S.W.O.R.D. detector or
ensemble) — so this wraps `BYTETracker` behind a plain numpy interface.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _Detections:
    """Minimal stand-in for an Ultralytics Boxes object: what BYTETracker reads (conf, xywh, cls, boolean indexing)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Detections":
        return _Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])


class ByteTrackWrapper:
    """
    ByteTrack with the Ultralytics defaults (bytetrack.yaml).

    `track_buffer` is in frames: keep it ≈ fps × seconds-of-occlusion-to-survive. Low-confidence detections
    (between track_low_thresh and track_high_thresh) are only used to continue existing tracks, which is why
    detectors should be run with a low confidence threshold (≈0.1) when feeding the tracker.
    """

    def __init__(self, track_high_thresh: float = 0.25, track_low_thresh: float = 0.1, new_track_thresh: float = 0.25,
                 track_buffer: int = 30, match_thresh: float = 0.8, fuse_score: bool = True):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.cfg = dict(track_high_thresh=track_high_thresh, track_low_thresh=track_low_thresh,
                        new_track_thresh=new_track_thresh, track_buffer=track_buffer, match_thresh=match_thresh,
                        fuse_score=fuse_score)
        self._tracker = BYTETracker(SimpleNamespace(tracker_type="bytetrack", **self.cfg))

    def update(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray | None = None) -> list[dict]:
        """Feed one frame of detections; returns the confirmed tracks visible in this frame."""
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        cls = np.zeros_like(conf) if cls is None else cls
        out = self._tracker.update(_Detections(xyxy, conf, cls))
        tracks = []
        for x1, y1, x2, y2, tid, score, c, det_idx in np.asarray(out).reshape(-1, 8):
            tracks.append({"track_id": int(tid), "bbox": [float(x1), float(y1), float(x2), float(y2)],
                           "score": float(score), "cls": int(c), "det_idx": int(det_idx)})
        return tracks

    def reset(self) -> None:
        self._tracker.reset()

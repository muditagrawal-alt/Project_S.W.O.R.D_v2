"""
Phase 2 interface — the contract a person-tracking module must satisfy to plug into the S.W.O.R.D. pipeline.

Phase 2 (persistent tracking + trajectory features) is owned by a teammate and will replace `StandInPhase2`.
Any object with this `update` method works; pass it as `SwordSystem(phase2=YourTracker(...))`.

    update(frame_bgr, t) -> (people, zone_events)
        people: list of dicts, one per tracked person visible in this frame
            {"track_id": int,                 # persistent across frames (required: Phases 3-5 rely on it)
             "bbox": [x1, y1, x2, y2],        # pixels
             "score": float,                  # detection confidence
             "trajectory": dict | None}       # optional; keys read downstream (any may be missing):
                                              #   speed_heights_s (body heights / s), zone (str | None),
                                              #   velocity_px_s, acceleration_px_s2, heading_deg, dwell_s,
                                              #   speed_m_s (only with a calibrated homography)
        zone_events: list of {"type": "zone_enter" | "zone_exit", "track_id": int, "zone": str, "timestamp": float}
    reset() -> None                          # called at the start of every new video/stream

`StandInPhase2` below (COCO YOLO26n person detector + Ultralytics ByteTrack + tracking.trajectory) exists only so
the end-to-end system can be built and tested now; it is not the final Phase 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

from tracking.tracker import ByteTrackWrapper
from tracking.trajectory import TrajectoryAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSON_MODEL = PROJECT_ROOT / "yolo26m" / "resumed_run" / "weights" / "yolo26n.pt"  # COCO-pretrained YOLO26n


class Phase2Tracker(Protocol):
    def update(self, frame_bgr: np.ndarray, t: float) -> Tuple[List[dict], List[dict]]: ...

    def reset(self) -> None: ...


class StandInPhase2:
    """Temporary Phase 2: YOLO26n (COCO person class) -> ByteTrack -> trajectory/zone features."""

    def __init__(self, fps: float, device: str, person_model: Path = DEFAULT_PERSON_MODEL, det_conf: float = 0.1,
                 track_seconds: float = 1.5, zones: Optional[list] = None, homography=None):
        from ultralytics import YOLO

        self.fps, self.device, self.det_conf = fps, device, det_conf
        self.track_buffer = max(5, int(round(fps * track_seconds)))
        self.detector = YOLO(str(person_model))
        self.zones, self.homography = zones, homography
        self.reset()
        self.name = f"stand-in: {Path(person_model).name} + ByteTrack + trajectory"

    def reset(self) -> None:
        self.tracker = ByteTrackWrapper(track_buffer=self.track_buffer)
        self.traj = TrajectoryAnalyzer(zones=self.zones, homography=self.homography)

    def update(self, frame_bgr: np.ndarray, t: float) -> Tuple[List[dict], List[dict]]:
        r = self.detector.predict(frame_bgr, conf=self.det_conf, classes=[0], device=self.device, verbose=False)[0]
        people = self.tracker.update(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
        feats, events = self.traj.update(t, people)
        return [{**p, "trajectory": feats.get(p["track_id"])} for p in people], events

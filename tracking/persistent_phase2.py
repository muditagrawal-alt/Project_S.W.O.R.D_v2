"""
Phase 2 from the team's persistent_video_tracking package, behind the Phase2Tracker interface (tracking/interface.py).

People are detected by the package's PersonDetector (COCO person class) and tracked by the person stream of its
DualStreamTracker (supervision ByteTrack, an ID namespace of its own). Per-track trajectory features for Phase 5
(footpoint velocity, body-height-normalised speed, heading, dwell, zones) come from tracking.trajectory, as for the
stand-in, so the fusion rules read the same keys. The package's video-level analytics (trajectory segments,
kinematics, zone events and dwell episodes) are available from finalize().

    system = SwordSystem(fps=fps, phase2=PersistentPhase2(fps, device))

legacy=True reproduces the package as submitted (person detections cut at 0.35 before tracking, trackers not
updated on frames without detections, buffer not adapted to the frame rate); it exists only for benchmarking.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tracking.interface import DEFAULT_PERSON_MODEL
from tracking.trajectory import TrajectoryAnalyzer


class PersistentPhase2:
    def __init__(self, fps: float, device: Optional[str] = None, person_model: Path = DEFAULT_PERSON_MODEL,
                 det_conf: float = 0.1, activation_thr: float = 0.30, lost_track_buffer: int = 30,
                 matching_thr: float = 0.75, img_size: int = 640, zones: Optional[list] = None, homography=None,
                 legacy: bool = False, detector=None):
        self.fps, self.legacy = float(fps), bool(legacy)
        self.zones, self.homography = zones, homography
        self.tracker_kwargs = dict(person_activation_thr=activation_thr, person_lost_buffer=lost_track_buffer,
                                   person_matching_thr=matching_thr,
                                   frame_rate=30.0 if legacy else self.fps, update_on_empty=not legacy)
        if detector is None:
            from persistent_video_tracking.detection.person_detector import PersonDetector

            detector = PersonDetector(weights_path=str(person_model), conf_thr=0.35 if legacy else det_conf,
                                      img_size=img_size, device=device)
        self.detector = detector
        self.name = (f"persistent_video_tracking{' (as submitted)' if legacy else ''}: "
                     f"{Path(str(person_model)).name} + supervision ByteTrack + trajectory")
        self.reset()

    def reset(self) -> None:
        import supervision as sv
        from persistent_video_tracking.tracking.dual_tracker import DualStreamTracker

        self._empty = sv.Detections.empty()
        self.tracker = DualStreamTracker(**self.tracker_kwargs)
        self.traj = TrajectoryAnalyzer(zones=self.zones, homography=self.homography)
        self.observations: list = []
        self.frame_id = 0

    def update(self, frame_bgr: np.ndarray, t: float) -> Tuple[List[dict], List[dict]]:
        obs, _, _ = self.tracker.update(weapon_detections=self._empty, person_detections=self.detector.detect(frame_bgr),
                                        video_name="stream", frame_id=self.frame_id, timestamp_seconds=float(t))
        self.frame_id += 1
        self.observations.extend(obs)
        people = [{"track_id": int(o.track_id), "bbox": [float(o.x1), float(o.y1), float(o.x2), float(o.y2)],
                   "score": float(o.confidence)} for o in obs if o.object_type == "person"]
        feats, events = self.traj.update(t, people)
        return [{**p, "trajectory": feats.get(p["track_id"])} for p in people], events

    def finalize(self):
        """Video-level analytics of the package: (segments, kinematics, zone_events, dwell_episodes)."""
        from persistent_video_tracking.analytics import KinematicsEngine, ZoneDwellEngine

        segments, kinematics = KinematicsEngine().compute_kinematics(self.observations, fps=self.fps)
        zones = {"definitions": {z["name"]: z["polygon"] for z in (self.zones or [])}}
        zone_events, dwells = ZoneDwellEngine(zones_config=zones).analyze(self.observations, fps=self.fps)
        return segments, kinematics, zone_events, dwells

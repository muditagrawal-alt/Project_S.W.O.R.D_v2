"""
Dual-Stream ByteTrack Tracker for S.W.O.R.D Phase 2.
Maintains completely isolated tracking instances and ID namespaces for:
1. Weapon Tracks (object_type='weapon')
2. Person Tracks (object_type='person')
Includes temporal class smoothing for weapons.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import supervision as sv

try:
    from .schema import TrackObservation
except (ImportError, ValueError):
    from persistent_video_tracking.tracking.schema import TrackObservation



class DualStreamTracker:
    """
    Orchestrates two independent ByteTrack trackers for weapons and persons.
    Ensures no ID collisions, extracts footpoints for persons, and applies
    temporal class smoothing to weapon detections.
    """

    def __init__(
        self,
        weapon_activation_thr: float = 0.25,
        weapon_lost_buffer: int = 30,
        weapon_matching_thr: float = 0.8,
        person_activation_thr: float = 0.30,
        person_lost_buffer: int = 30,
        person_matching_thr: float = 0.75,
        weapon_smoothing_window: int = 15,
        weapon_class_names: Optional[Dict[int, str]] = None,
    ):
        # 1. Weapon ByteTrack Instance
        self.weapon_tracker = sv.ByteTrack(
            track_activation_threshold=float(weapon_activation_thr),
            lost_track_buffer=int(weapon_lost_buffer),
            minimum_matching_threshold=float(weapon_matching_thr),
        )

        # 2. Person ByteTrack Instance (Completely separate namespace and internal state)
        self.person_tracker = sv.ByteTrack(
            track_activation_threshold=float(person_activation_thr),
            lost_track_buffer=int(person_lost_buffer),
            minimum_matching_threshold=float(person_matching_thr),
        )

        self.weapon_class_names = weapon_class_names or {}
        self.weapon_smoothing_window = weapon_smoothing_window

        # Sliding window history for weapon class smoothing: track_id -> deque of (class_name, confidence)
        self.weapon_class_history = defaultdict(lambda: deque(maxlen=self.weapon_smoothing_window))

    def update(
        self,
        weapon_detections: sv.Detections,
        person_detections: sv.Detections,
        video_name: str,
        frame_id: int,
        timestamp_seconds: float,
    ) -> Tuple[List[TrackObservation], sv.Detections, sv.Detections]:
        """
        Update both tracking streams with frame detections.

        Args:
            weapon_detections: Raw fused weapon detections from WeaponEnsembleDetector.
            person_detections: Raw person detections from PersonDetector.
            video_name: Source video filename.
            frame_id: 0-indexed or 1-indexed video frame counter.
            timestamp_seconds: Actual elapsed time in seconds.

        Returns:
            Tuple of:
                - List of TrackObservation objects for this frame
                - Updated weapon sv.Detections (with tracker_id)
                - Updated person sv.Detections (with tracker_id)
        """
        observations: List[TrackObservation] = []

        # --- A. WEAPON TRACKING STREAM ---
        tracked_weapons = sv.Detections.empty()
        if len(weapon_detections) > 0:
            tracked_weapons = self.weapon_tracker.update_with_detections(weapon_detections)

        if len(tracked_weapons) > 0 and tracked_weapons.tracker_id is not None:
            for i in range(len(tracked_weapons)):
                tid = int(tracked_weapons.tracker_id[i])
                cid = int(tracked_weapons.class_id[i])
                conf = float(tracked_weapons.confidence[i])
                x1, y1, x2, y2 = map(float, tracked_weapons.xyxy[i])

                raw_class_name = self.weapon_class_names.get(cid, f"weapon_{cid}")

                # Temporal class smoothing
                self.weapon_class_history[tid].append((raw_class_name, conf))
                smoothed_class = self._compute_smoothed_class(tid)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bcx = cx
                bcy = y2  # bottom center

                obs = TrackObservation(
                    video_name=video_name,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    object_type="weapon",
                    track_id=tid,
                    class_name=smoothed_class,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    center_x=cx,
                    center_y=cy,
                    bottom_center_x=bcx,
                    bottom_center_y=bcy,
                )
                observations.append(obs)

        # --- B. PERSON TRACKING STREAM ---
        tracked_persons = sv.Detections.empty()
        if len(person_detections) > 0:
            tracked_persons = self.person_tracker.update_with_detections(person_detections)

        if len(tracked_persons) > 0 and tracked_persons.tracker_id is not None:
            for i in range(len(tracked_persons)):
                pid = int(tracked_persons.tracker_id[i])
                conf = float(tracked_persons.confidence[i])
                x1, y1, x2, y2 = map(float, tracked_persons.xyxy[i])

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                # Footpoint / bottom-center for spatial ground contact
                foot_x = cx
                foot_y = y2

                obs = TrackObservation(
                    video_name=video_name,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    object_type="person",
                    track_id=pid,
                    class_name="person",
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    center_x=cx,
                    center_y=cy,
                    bottom_center_x=foot_x,
                    bottom_center_y=foot_y,
                )
                observations.append(obs)

        return observations, tracked_weapons, tracked_persons

    def _compute_smoothed_class(self, track_id: int) -> str:
        """Confidence-weighted voting over sliding window history."""
        history = self.weapon_class_history[track_id]
        if not history:
            return "weapon"

        scores: Dict[str, float] = defaultdict(float)
        for cls_name, conf in history:
            scores[cls_name] += conf

        return max(scores.items(), key=lambda x: x[1])[0]

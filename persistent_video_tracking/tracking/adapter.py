"""
Public Integration Adapter for S.W.O.R.D. v2 Persistent Video Tracking.
Provides a clean, standardized programmatic interface for integrating persistent
video tracking into the central S.W.O.R.D. application, live RTSP feeds, Streamlit dashboard,
or downstream pose estimation and threat scoring pipelines.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import yaml
import supervision as sv

try:
    from ..detection.weapon_ensemble import WeaponEnsembleDetector
    from ..detection.person_detector import PersonDetector
    from .schema import TrackObservation, TRACKING_CSV_COLUMNS
    from .dual_tracker import DualStreamTracker
    from ..association.person_weapon_matcher import SpatioTemporalAssociator, AssociationRecord
    from ..analytics.zones_dwell import ZoneDwellEngine, ZoneEvent, DwellEpisode
    from ..analytics.kinematics import KinematicsEngine, TrajectorySegment, KinematicObservation
except (ImportError, ValueError):
    from persistent_video_tracking.detection.weapon_ensemble import WeaponEnsembleDetector
    from persistent_video_tracking.detection.person_detector import PersonDetector
    from persistent_video_tracking.tracking.schema import TrackObservation, TRACKING_CSV_COLUMNS
    from persistent_video_tracking.tracking.dual_tracker import DualStreamTracker
    from persistent_video_tracking.association.person_weapon_matcher import SpatioTemporalAssociator, AssociationRecord
    from persistent_video_tracking.analytics.zones_dwell import ZoneDwellEngine, ZoneEvent, DwellEpisode
    from persistent_video_tracking.analytics.kinematics import KinematicsEngine, TrajectorySegment, KinematicObservation


@dataclass
class FrameTrackingResult:
    """Standardized output structure for a single processed video frame."""
    frame_id: int
    timestamp_seconds: float
    weapon_tracks: List[TrackObservation]
    person_tracks: List[TrackObservation]
    associations: List[AssociationRecord]
    raw_weapon_detections: sv.Detections
    raw_person_detections: sv.Detections


class PersistentVideoTracker:
    """
    Standard integration adapter for S.W.O.R.D. v2.
    Orchestrates the entire multi-model tracking and analytics subsystem
    for synchronous frame-by-frame processing or video-level analysis.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ):
        if config_dict is not None:
            self.cfg = config_dict
        elif config_path is not None:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            # Default configuration in package
            default_cfg_path = Path(__file__).resolve().parent.parent / "config" / "tracking_config.yaml"
            with open(default_cfg_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        dev = device or self.cfg.get("models", {}).get("device", "cpu")
        det_cfg = self.cfg.get("detection", {})
        w_det_cfg = det_cfg.get("weapon", {})
        p_det_cfg = det_cfg.get("person", {})
        models_cfg = self.cfg.get("models", {})

        # 1. Weapon Ensemble
        self.weapon_detector = WeaponEnsembleDetector(
            weights_s=models_cfg.get("weapon_s", "weights/yolo26s_best.pt"),
            weights_m=models_cfg.get("weapon_m", "weights/yolo26m_best.pt"),
            model_weights=tuple(w_det_cfg.get("wbf_weights", [1.0, 1.1])),
            conf_thr=float(w_det_cfg.get("conf_threshold", 0.25)),
            iou_thr=float(w_det_cfg.get("iou_threshold", 0.55)),
            img_size=int(det_cfg.get("img_size", 640)),
            fusion_mode=w_det_cfg.get("fusion_mode", "wbf"),
            conf_type=w_det_cfg.get("conf_type", "avg"),
            device=dev,
        )

        # 2. Person Detector
        self.person_detector = PersonDetector(
            weights_path=models_cfg.get("person", "yolov8n.pt"),
            conf_thr=float(p_det_cfg.get("conf_threshold", 0.35)),
            img_size=int(det_cfg.get("img_size", 640)),
            device=dev,
            coco_person_class_id=int(p_det_cfg.get("coco_person_class_id", 0)),
        )

        # 3. Dual-Stream Tracker
        tr_cfg = self.cfg.get("tracking", {})
        w_tr = tr_cfg.get("weapon", {})
        p_tr = tr_cfg.get("person", {})

        self.dual_tracker = DualStreamTracker(
            weapon_activation_thr=float(w_tr.get("track_activation_threshold", 0.25)),
            weapon_lost_buffer=int(w_tr.get("lost_track_buffer", 30)),
            weapon_matching_thr=float(w_tr.get("minimum_matching_threshold", 0.80)),
            person_activation_thr=float(p_tr.get("track_activation_threshold", 0.30)),
            person_lost_buffer=int(p_tr.get("lost_track_buffer", 30)),
            person_matching_thr=float(p_tr.get("minimum_matching_threshold", 0.75)),
            weapon_class_names=self.weapon_detector.class_names,
        )

        # 4. Association Engine
        assoc_cfg = self.cfg.get("association", {})
        self.associator = SpatioTemporalAssociator(
            iow_threshold=float(assoc_cfg.get("iow_threshold", 0.15)),
            proximity_distance_px=float(assoc_cfg.get("proximity_distance_px", 120.0)),
            temporal_window_frames=int(assoc_cfg.get("temporal_window_frames", 5)),
            min_association_score=float(assoc_cfg.get("min_association_score", 0.35)),
        )

        # 5. Zone & Kinematics Engines
        an_cfg = self.cfg.get("analytics", {})
        self.zone_engine = ZoneDwellEngine(
            zones_config=self.cfg.get("zones", {}),
            frame_width=frame_width,
            frame_height=frame_height,
            dwell_radius_px=float(an_cfg.get("dwell_radius_px", 50.0)),
            minimum_dwell_seconds=float(an_cfg.get("minimum_dwell_seconds", 1.0)),
            stationary_threshold_px_s=float(an_cfg.get("stationary_velocity_threshold_px_s", 15.0)),
        )

        self.kinematics_engine = KinematicsEngine(
            stationary_threshold_px_s=float(an_cfg.get("stationary_velocity_threshold_px_s", 15.0)),
            min_segment_length=int(an_cfg.get("min_segment_length", 5)),
            max_continuous_gap=int(an_cfg.get("max_continuous_frame_gap", 1)),
        )

        self.accumulated_observations: List[TrackObservation] = []
        self.accumulated_associations: List[AssociationRecord] = []

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_seconds: float,
        video_name: str = "stream",
    ) -> FrameTrackingResult:
        """
        Process a single video frame: detection, dual tracking, and spatial association.
        """
        # A. Detect Weapons (Ensemble WBF)
        w_dets = self.weapon_detector.detect(frame)

        # B. Detect Persons (YOLOv8n)
        p_dets = self.person_detector.detect(frame)

        # C. Update Dual Tracker
        frame_obs, tracked_w, tracked_p = self.dual_tracker.update(
            weapon_detections=w_dets,
            person_detections=p_dets,
            video_name=video_name,
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
        )
        self.accumulated_observations.extend(frame_obs)

        w_obs = [o for o in frame_obs if o.object_type == "weapon"]
        p_obs = [o for o in frame_obs if o.object_type == "person"]

        # D. Associate Weapons and Persons
        assocs = self.associator.associate_frame(
            video_name=video_name,
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            weapon_obs_list=w_obs,
            person_obs_list=p_obs,
        )
        self.accumulated_associations.extend(assocs)

        return FrameTrackingResult(
            frame_id=frame_id,
            timestamp_seconds=timestamp_seconds,
            weapon_tracks=w_obs,
            person_tracks=p_obs,
            associations=assocs,
            raw_weapon_detections=tracked_w,
            raw_person_detections=tracked_p,
        )

    def finalize(
        self,
        fps: float,
    ) -> Tuple[List[TrajectorySegment], List[KinematicObservation], List[ZoneEvent], List[DwellEpisode]]:
        """
        Finalize video-level motion analytics and zone dwell calculations.
        """
        segments, kinematics = self.kinematics_engine.compute_kinematics(
            observations=self.accumulated_observations,
            fps=fps,
        )
        zone_events, dwell_episodes = self.zone_engine.analyze(
            observations=self.accumulated_observations,
            fps=fps,
        )
        return segments, kinematics, zone_events, dwell_episodes

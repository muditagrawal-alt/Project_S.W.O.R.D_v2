"""
S.W.O.R.D. v2 — Persistent Video Tracking Feature Package.
A YOLO Ensemble and ByteTrack-Based Tracking and Motion Analytics Module.
"""

from .detection import WeaponEnsembleDetector, PersonDetector, SWORD_WEAPON_CLASSES, weighted_boxes_fusion
from .tracking import TrackObservation, TRACKING_CSV_COLUMNS, DualStreamTracker, PersistentVideoTracker, FrameTrackingResult
from .analytics import KinematicsEngine, TrajectorySegment, KinematicObservation, ZoneDwellEngine, ZoneEvent, DwellEpisode
from .association import SpatioTemporalAssociator, AssociationRecord, compute_iow_and_distance
from .visualization import VideoVisualizer
from .pipeline import run_pipeline, load_config, PersistentTrackingValidator, validate_run

__version__ = "2.0.0"

__all__ = [
    # Detection
    "WeaponEnsembleDetector",
    "PersonDetector",
    "SWORD_WEAPON_CLASSES",
    "weighted_boxes_fusion",
    # Tracking
    "TrackObservation",
    "TRACKING_CSV_COLUMNS",
    "DualStreamTracker",
    "PersistentVideoTracker",
    "FrameTrackingResult",
    # Analytics
    "KinematicsEngine",
    "TrajectorySegment",
    "KinematicObservation",
    "ZoneDwellEngine",
    "ZoneEvent",
    "DwellEpisode",
    # Association
    "SpatioTemporalAssociator",
    "AssociationRecord",
    "compute_iow_and_distance",
    # Visualization
    "VideoVisualizer",
    # Pipeline
    "run_pipeline",
    "load_config",
    "PersistentTrackingValidator",
    "validate_run",
]

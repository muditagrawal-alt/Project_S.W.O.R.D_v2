"""
Generic Track Data Schema for S.W.O.R.D Phase 2.
Standardized record representation for both weapon and person tracking streams.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List


# Standard CSV columns for unified tracking observations
TRACKING_CSV_COLUMNS = [
    "video_name",
    "frame_id",
    "timestamp_seconds",
    "object_type",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "bottom_center_x",
    "bottom_center_y",
]


@dataclass
class TrackObservation:
    video_name: str
    frame_id: int
    timestamp_seconds: float
    object_type: str  # "weapon" or "person"
    track_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    bottom_center_x: float
    bottom_center_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "frame_id": self.frame_id,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4),
            "object_type": self.object_type,
            "track_id": int(self.track_id),
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "x1": round(float(self.x1), 2),
            "y1": round(float(self.y1), 2),
            "x2": round(float(self.x2), 2),
            "y2": round(float(self.y2), 2),
            "center_x": round(float(self.center_x), 2),
            "center_y": round(float(self.center_y), 2),
            "bottom_center_x": round(float(self.bottom_center_x), 2),
            "bottom_center_y": round(float(self.bottom_center_y), 2),
        }

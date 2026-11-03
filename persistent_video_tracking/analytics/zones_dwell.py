"""
Zone Analytics and Dwell Time Engine for S.W.O.R.D Phase 2.
Implements:
1. Configurable polygon zones using cv2.pointPolygonTest (zero extra dependencies).
2. Person ground contact using bottom-center (footpoint); weapon centroid for weapons.
3. Zone event detection: ZONE_ENTRY, ZONE_EXIT, ZONE_TRANSITION.
4. Zone dwell time calculation (consecutive and cumulative time spent inside a zone).
5. Localized stationary dwell episodes (confined within dwell_radius or stationary speed).
Gracefully handles unconfigured/empty zones.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import cv2

try:
    from ..tracking.schema import TrackObservation
except (ImportError, ValueError):
    from persistent_video_tracking.tracking.schema import TrackObservation



@dataclass
class ZoneEvent:
    video_name: str
    object_type: str
    track_id: int
    from_zone: Optional[str]
    to_zone: Optional[str]
    event: str  # "ZONE_ENTRY", "ZONE_EXIT", "ZONE_TRANSITION"
    frame_id: int
    timestamp_seconds: float
    dwell_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "object_type": self.object_type,
            "track_id": self.track_id,
            "from_zone": self.from_zone or "OUTSIDE",
            "to_zone": self.to_zone or "OUTSIDE",
            "event": self.event,
            "frame_id": self.frame_id,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4),
            "dwell_seconds": round(float(self.dwell_seconds), 4),
        }


@dataclass
class DwellEpisode:
    video_name: str
    object_type: str
    track_id: int
    dwell_type: str  # "ZONE_BASED" or "LOCALIZED_STATIONARY"
    zone_name: Optional[str]
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_seconds: float
    anchor_x: float
    anchor_y: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "object_type": self.object_type,
            "track_id": self.track_id,
            "dwell_type": self.dwell_type,
            "zone_name": self.zone_name or "N/A",
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": round(float(self.start_time_s), 4),
            "end_time_s": round(float(self.end_time_s), 4),
            "duration_seconds": round(float(self.duration_seconds), 4),
            "anchor_x": round(float(self.anchor_x), 2),
            "anchor_y": round(float(self.anchor_y), 2),
        }


class ZoneDwellEngine:
    """
    Manages zone containment testing and computes zone events and dwell times.
    """

    def __init__(
        self,
        zones_config: Optional[Dict[str, Any]] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        dwell_radius_px: float = 50.0,
        minimum_dwell_seconds: float = 1.0,
        stationary_threshold_px_s: float = 15.0,
    ):
        self.dwell_radius_px = float(dwell_radius_px)
        self.minimum_dwell_seconds = float(minimum_dwell_seconds)
        self.stationary_threshold_px_s = float(stationary_threshold_px_s)
        self.zone_polygons: Dict[str, np.ndarray] = {}

        if zones_config and "definitions" in zones_config:
            is_norm = zones_config.get("normalized", False)
            defs = zones_config.get("definitions", {})

            for zone_name, pts in defs.items():
                if not pts or len(pts) < 3:
                    continue
                arr = np.array(pts, dtype=np.float32)
                if is_norm and frame_width and frame_height:
                    arr[:, 0] *= frame_width
                    arr[:, 1] *= frame_height
                self.zone_polygons[zone_name] = arr

        print(f"[ZoneDwellEngine] Loaded {len(self.zone_polygons)} configured zones.", flush=True)

    def find_zone(self, x: float, y: float) -> Optional[str]:
        """Test point against all polygon zones; returns zone name or None."""
        pt = (float(x), float(y))
        for zone_name, poly in self.zone_polygons.items():
            # cv2.pointPolygonTest >= 0 means inside or on edge
            if cv2.pointPolygonTest(poly, pt, measureDist=False) >= 0:
                return zone_name
        return None

    def analyze(
        self,
        observations: List[TrackObservation],
        fps: float,
    ) -> Tuple[List[ZoneEvent], List[DwellEpisode]]:
        """
        Analyze zone transitions and localized stationary dwell for all tracks.

        Args:
            observations: List of TrackObservation objects for the video.
            fps: Video frames per second.

        Returns:
            Tuple of:
                - List of ZoneEvent records
                - List of DwellEpisode records
        """
        if not observations or fps <= 0:
            return [], []

        # Group observations by (object_type, track_id)
        grouped = defaultdict(list)
        for obs in observations:
            grouped[(obs.object_type, obs.track_id)].append(obs)

        all_events: List[ZoneEvent] = []
        all_dwells: List[DwellEpisode] = []

        for (obj_type, tid), track_obs in grouped.items():
            track_obs.sort(key=lambda o: o.frame_id)

            # ------------------------------------------------------------------
            # 1. Zone Event & Zone Dwell Analysis (if zones exist)
            # ------------------------------------------------------------------
            if self.zone_polygons:
                current_zone: Optional[str] = None
                zone_entry_frame: Optional[int] = None
                zone_entry_time: Optional[float] = None

                for obs in track_obs:
                    # Person uses bottom_center (footpoint); weapon uses centroid
                    anchor_x = obs.bottom_center_x if obj_type == "person" else obs.center_x
                    anchor_y = obs.bottom_center_y if obj_type == "person" else obs.center_y

                    detected_zone = self.find_zone(anchor_x, anchor_y)

                    if detected_zone != current_zone:
                        # Zone state changed
                        if current_zone is not None and detected_zone is None:
                            # ZONE_EXIT
                            dwell_s = (obs.frame_id - zone_entry_frame) / fps if zone_entry_frame is not None else 0.0
                            all_events.append(ZoneEvent(
                                video_name=obs.video_name,
                                object_type=obj_type,
                                track_id=tid,
                                from_zone=current_zone,
                                to_zone="OUTSIDE",
                                event="ZONE_EXIT",
                                frame_id=obs.frame_id,
                                timestamp_seconds=obs.timestamp_seconds,
                                dwell_seconds=dwell_s,
                            ))
                            if dwell_s >= self.minimum_dwell_seconds:
                                all_dwells.append(DwellEpisode(
                                    video_name=obs.video_name,
                                    object_type=obj_type,
                                    track_id=tid,
                                    dwell_type="ZONE_BASED",
                                    zone_name=current_zone,
                                    start_frame=zone_entry_frame,
                                    end_frame=obs.frame_id,
                                    start_time_s=zone_entry_time,
                                    end_time_s=obs.timestamp_seconds,
                                    duration_seconds=dwell_s,
                                    anchor_x=anchor_x,
                                    anchor_y=anchor_y,
                                ))
                            current_zone = None
                            zone_entry_frame = None
                            zone_entry_time = None

                        elif current_zone is None and detected_zone is not None:
                            # ZONE_ENTRY
                            current_zone = detected_zone
                            zone_entry_frame = obs.frame_id
                            zone_entry_time = obs.timestamp_seconds
                            all_events.append(ZoneEvent(
                                video_name=obs.video_name,
                                object_type=obj_type,
                                track_id=tid,
                                from_zone="OUTSIDE",
                                to_zone=detected_zone,
                                event="ZONE_ENTRY",
                                frame_id=obs.frame_id,
                                timestamp_seconds=obs.timestamp_seconds,
                                dwell_seconds=0.0,
                            ))

                        elif current_zone is not None and detected_zone is not None:
                            # ZONE_TRANSITION (directly from Zone A to Zone B)
                            dwell_s = (obs.frame_id - zone_entry_frame) / fps if zone_entry_frame is not None else 0.0
                            all_events.append(ZoneEvent(
                                video_name=obs.video_name,
                                object_type=obj_type,
                                track_id=tid,
                                from_zone=current_zone,
                                to_zone=detected_zone,
                                event="ZONE_TRANSITION",
                                frame_id=obs.frame_id,
                                timestamp_seconds=obs.timestamp_seconds,
                                dwell_seconds=dwell_s,
                            ))
                            if dwell_s >= self.minimum_dwell_seconds:
                                all_dwells.append(DwellEpisode(
                                    video_name=obs.video_name,
                                    object_type=obj_type,
                                    track_id=tid,
                                    dwell_type="ZONE_BASED",
                                    zone_name=current_zone,
                                    start_frame=zone_entry_frame,
                                    end_frame=obs.frame_id,
                                    start_time_s=zone_entry_time,
                                    end_time_s=obs.timestamp_seconds,
                                    duration_seconds=dwell_s,
                                    anchor_x=anchor_x,
                                    anchor_y=anchor_y,
                                ))
                            current_zone = detected_zone
                            zone_entry_frame = obs.frame_id
                            zone_entry_time = obs.timestamp_seconds

                # If track ended while still in a zone
                if current_zone is not None and zone_entry_frame is not None:
                    last_obs = track_obs[-1]
                    dwell_s = (last_obs.frame_id - zone_entry_frame + 1) / fps
                    if dwell_s >= self.minimum_dwell_seconds:
                        all_dwells.append(DwellEpisode(
                            video_name=last_obs.video_name,
                            object_type=obj_type,
                            track_id=tid,
                            dwell_type="ZONE_BASED",
                            zone_name=current_zone,
                            start_frame=zone_entry_frame,
                            end_frame=last_obs.frame_id,
                            start_time_s=zone_entry_time,
                            end_time_s=last_obs.timestamp_seconds,
                            duration_seconds=dwell_s,
                            anchor_x=last_obs.center_x,
                            anchor_y=last_obs.center_y,
                        ))

            # ------------------------------------------------------------------
            # 2. Localized / Stationary Dwell Analysis
            # ------------------------------------------------------------------
            # Detect periods where track stays within dwell_radius_px
            stat_anchor_obs = track_obs[0]
            stat_start_frame = track_obs[0].frame_id
            stat_start_time = track_obs[0].timestamp_seconds
            stat_count = 1

            for obs in track_obs[1:]:
                # Distance from current anchor
                dist = np.hypot(obs.center_x - stat_anchor_obs.center_x, obs.center_y - stat_anchor_obs.center_y)
                if dist <= self.dwell_radius_px:
                    stat_count += 1
                else:
                    duration_s = (obs.frame_id - stat_start_frame) / fps
                    if duration_s >= self.minimum_dwell_seconds and stat_count >= 5:
                        all_dwells.append(DwellEpisode(
                            video_name=obs.video_name,
                            object_type=obj_type,
                            track_id=tid,
                            dwell_type="LOCALIZED_STATIONARY",
                            zone_name=None,
                            start_frame=stat_start_frame,
                            end_frame=obs.frame_id - 1,
                            start_time_s=stat_start_time,
                            end_time_s=obs.timestamp_seconds,
                            duration_seconds=duration_s,
                            anchor_x=stat_anchor_obs.center_x,
                            anchor_y=stat_anchor_obs.center_y,
                        ))
                    # Reset anchor to current observation
                    stat_anchor_obs = obs
                    stat_start_frame = obs.frame_id
                    stat_start_time = obs.timestamp_seconds
                    stat_count = 1

            # Check final segment
            duration_s = (track_obs[-1].frame_id - stat_start_frame + 1) / fps
            if duration_s >= self.minimum_dwell_seconds and stat_count >= 5:
                all_dwells.append(DwellEpisode(
                    video_name=track_obs[-1].video_name,
                    object_type=obj_type,
                    track_id=tid,
                    dwell_type="LOCALIZED_STATIONARY",
                    zone_name=None,
                    start_frame=stat_start_frame,
                    end_frame=track_obs[-1].frame_id,
                    start_time_s=stat_start_time,
                    end_time_s=track_obs[-1].timestamp_seconds,
                    duration_seconds=duration_s,
                    anchor_x=stat_anchor_obs.center_x,
                    anchor_y=stat_anchor_obs.center_y,
                ))

        return all_events, all_dwells

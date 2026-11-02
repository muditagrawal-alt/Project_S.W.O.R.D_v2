"""
Person <-> Weapon Spatio-Temporal Association Engine for S.W.O.R.D Phase 2.
Estimates spatial-temporal association between weapon tracks and person tracks.
DOES NOT alter original track IDs in either stream.
Labelled strictly as estimated spatial-temporal association, not proven ownership.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import math
import numpy as np

try:
    from ..tracking.schema import TrackObservation
except (ImportError, ValueError):
    from persistent_video_tracking.tracking.schema import TrackObservation



@dataclass
class AssociationRecord:
    video_name: str
    frame_id: int
    timestamp_seconds: float
    weapon_track_id: int
    person_track_id: Optional[int]
    association_score: float
    association_method: str
    association_status: str  # "ASSOCIATED", "UNASSOCIATED"
    iow: float  # Intersection over Weapon area [0..1]
    distance_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "frame_id": self.frame_id,
            "timestamp_seconds": round(float(self.timestamp_seconds), 4),
            "weapon_track_id": self.weapon_track_id,
            "person_track_id": self.person_track_id if self.person_track_id is not None else -1,
            "association_score": round(float(self.association_score), 4),
            "association_method": self.association_method,
            "association_status": self.association_status,
            "iow": round(float(self.iow), 4),
            "distance_px": round(float(self.distance_px), 2),
        }


def compute_iow_and_distance(
    w_box: Tuple[float, float, float, float],
    p_box: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Compute Intersection over Weapon (IoW) and centroid-to-person distance.
    w_box, p_box: (x1, y1, x2, y2)
    """
    wx1, wy1, wx2, wy2 = w_box
    px1, py1, px2, py2 = p_box

    # Intersection
    ix1 = max(wx1, px1)
    iy1 = max(wy1, py1)
    ix2 = min(wx2, px2)
    iy2 = min(wy2, py2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    w_area = max(1.0, (wx2 - wx1) * (wy2 - wy1))
    iow = inter_area / w_area

    # Centroid distance
    wcx = (wx1 + wx2) / 2.0
    wcy = (wy1 + wy2) / 2.0
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    dist = math.hypot(wcx - pcx, wcy - pcy)

    return iow, dist


class SpatioTemporalAssociator:
    """
    Associates weapon tracks with person tracks using:
    1. Intersection-over-Weapon (IoW) containment
    2. Proximity distance
    3. Multi-frame temporal voting window to prevent flicker
    """

    def __init__(
        self,
        iow_threshold: float = 0.15,
        proximity_distance_px: float = 120.0,
        temporal_window_frames: int = 5,
        min_association_score: float = 0.35,
    ):
        self.iow_threshold = float(iow_threshold)
        self.proximity_distance_px = float(proximity_distance_px)
        self.temporal_window = int(temporal_window_frames)
        self.min_association_score = float(min_association_score)

        # Sliding window history of best candidate per weapon_track_id
        # weapon_track_id -> deque of (person_track_id, score)
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.temporal_window))

    def associate_frame(
        self,
        video_name: str,
        frame_id: int,
        timestamp_seconds: float,
        weapon_obs_list: List[TrackObservation],
        person_obs_list: List[TrackObservation],
    ) -> List[AssociationRecord]:
        """
        Produce association records for all weapons detected in this frame.

        Args:
            video_name: Video identifier.
            frame_id: Current frame counter.
            timestamp_seconds: Elapsed video seconds.
            weapon_obs_list: Weapon observations in this frame.
            person_obs_list: Person observations in this frame.

        Returns:
            List of AssociationRecord for each weapon.
        """
        records: List[AssociationRecord] = []

        if not weapon_obs_list:
            return records

        for w_obs in weapon_obs_list:
            wid = w_obs.track_id
            w_box = (w_obs.x1, w_obs.y1, w_obs.x2, w_obs.y2)

            best_pid: Optional[int] = None
            best_score = 0.0
            best_iow = 0.0
            best_dist = float("inf")

            if person_obs_list:
                for p_obs in person_obs_list:
                    pid = p_obs.track_id
                    p_box = (p_obs.x1, p_obs.y1, p_obs.x2, p_obs.y2)

                    iow, dist = compute_iow_and_distance(w_box, p_box)

                    # Normalized proximity score [0..1]
                    prox_score = max(0.0, 1.0 - (dist / self.proximity_distance_px))

                    # Combined score: weighted fusion of overlap and proximity
                    score = 0.65 * iow + 0.35 * prox_score

                    if score > best_score:
                        best_score = score
                        best_pid = pid
                        best_iow = iow
                        best_dist = dist

            # Record into temporal history
            if best_pid is not None and best_score >= (self.min_association_score * 0.7):
                self.history[wid].append((best_pid, best_score))
            else:
                self.history[wid].append((None, 0.0))

            # Apply temporal voting window
            recent = self.history[wid]
            person_votes: Dict[Optional[int], float] = defaultdict(float)
            for r_pid, r_score in recent:
                if r_pid is not None:
                    person_votes[r_pid] += r_score

            if person_votes:
                top_voted_pid, total_accum_score = max(person_votes.items(), key=lambda x: x[1])
                avg_window_score = total_accum_score / len(recent)

                # Check if current best or temporal window surpasses threshold
                if best_score >= self.min_association_score or avg_window_score >= self.min_association_score:
                    chosen_pid = best_pid if (best_pid is not None and best_score >= self.min_association_score) else top_voted_pid
                    final_score = max(best_score, avg_window_score)
                    status = "ASSOCIATED"
                    method = "SPATIAL_IOW_PROXIMITY_TEMPORAL"
                else:
                    chosen_pid = None
                    final_score = best_score
                    status = "UNASSOCIATED"
                    method = "INSUFFICIENT_EVIDENCE"
            else:
                chosen_pid = None
                final_score = 0.0
                status = "UNASSOCIATED"
                method = "NO_PERSON_DETECTED"

            records.append(AssociationRecord(
                video_name=video_name,
                frame_id=frame_id,
                timestamp_seconds=timestamp_seconds,
                weapon_track_id=wid,
                person_track_id=chosen_pid,
                association_score=final_score,
                association_method=method,
                association_status=status,
                iow=best_iow,
                distance_px=best_dist if not math.isinf(best_dist) else 0.0,
            ))

        return records

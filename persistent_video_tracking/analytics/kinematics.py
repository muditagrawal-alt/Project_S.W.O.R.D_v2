"""
Kinematics and Trajectory Analytics for S.W.O.R.D Phase 2.
Implements:
1. Trajectory segmentation on frame_gap > 1 (preserving raw observations, zero gap-interpolation).
2. Direction analysis in image coordinates (heading degrees, 8-way compass, turn angle, stationary threshold).
3. Pixel-space velocity (vx, vy, speed in px/s, FPS-aware).
4. Acceleration (ax, ay, |a| in px/s^2 inside continuous segments).
Enforces N observations -> N-1 velocity steps -> N-2 acceleration steps.
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

try:
    from ..tracking.schema import TrackObservation
except (ImportError, ValueError):
    from persistent_video_tracking.tracking.schema import TrackObservation



@dataclass
class TrajectorySegment:
    segment_id: str
    video_name: str
    object_type: str
    track_id: int
    class_name: str
    observations_count: int
    start_frame: int
    end_frame: int
    duration_seconds: float
    is_valid_motion: bool  # True if observations_count >= min_segment_length
    total_path_length_px: float
    displacement_px: float


@dataclass
class KinematicObservation:
    video_name: str
    object_type: str
    track_id: int
    segment_id: str
    segment_classification: str
    from_frame: int
    to_frame: int
    from_time_s: float
    to_time_s: float
    dt_s: float
    from_x: float
    from_y: float
    to_x: float
    to_y: float
    dx_px: float
    dy_px: float
    step_distance_px: float
    # Velocity (px/s)
    vx_px_s: float
    vy_px_s: float
    speed_px_s: float
    # Direction
    is_stationary: bool
    heading_deg: Optional[float]
    direction_8way: str
    turn_angle_deg: Optional[float]
    # Acceleration (px/s^2) — available on step >= 2 of continuous segment
    ax_px_s2: Optional[float]
    ay_px_s2: Optional[float]
    acceleration_px_s2: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_name": self.video_name,
            "object_type": self.object_type,
            "track_id": self.track_id,
            "segment_id": self.segment_id,
            "segment_classification": self.segment_classification,
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "from_time_s": round(self.from_time_s, 4),
            "to_time_s": round(self.to_time_s, 4),
            "dt_s": round(self.dt_s, 5),
            "from_x": round(self.from_x, 2),
            "from_y": round(self.from_y, 2),
            "to_x": round(self.to_x, 2),
            "to_y": round(self.to_y, 2),
            "dx_px": round(self.dx_px, 2),
            "dy_px": round(self.dy_px, 2),
            "step_distance_px": round(self.step_distance_px, 2),
            "vx_px_s": round(self.vx_px_s, 2),
            "vy_px_s": round(self.vy_px_s, 2),
            "speed_px_s": round(self.speed_px_s, 2),
            "is_stationary": self.is_stationary,
            "heading_deg": round(self.heading_deg, 2) if self.heading_deg is not None else None,
            "direction_8way": self.direction_8way,
            "turn_angle_deg": round(self.turn_angle_deg, 2) if self.turn_angle_deg is not None else None,
            "ax_px_s2": round(self.ax_px_s2, 2) if self.ax_px_s2 is not None else None,
            "ay_px_s2": round(self.ay_px_s2, 2) if self.ay_px_s2 is not None else None,
            "acceleration_px_s2": round(self.acceleration_px_s2, 2) if self.acceleration_px_s2 is not None else None,
        }


class KinematicsEngine:
    """
    Computes trajectory segments, direction, velocity, and acceleration
    strictly respecting tracking boundaries and continuous observation segments.
    """

    COMPASS_SECTORS = [
        ("E", 0.0, 22.5),
        ("SE", 22.5, 67.5),
        ("S", 67.5, 112.5),
        ("SW", 112.5, 157.5),
        ("W", 157.5, 202.5),
        ("NW", 202.5, 247.5),
        ("N", 247.5, 292.5),
        ("NE", 292.5, 337.5),
        ("E", 337.5, 360.0),
    ]

    def __init__(
        self,
        stationary_threshold_px_s: float = 15.0,
        min_segment_length: int = 5,
        max_continuous_gap: int = 1,
    ):
        self.stationary_threshold = float(stationary_threshold_px_s)
        self.min_segment_length = int(min_segment_length)
        self.max_continuous_gap = int(max_continuous_gap)

    def heading_to_8way(self, heading_deg: Optional[float]) -> str:
        """Map image coordinate heading (0..360 deg) to 8-way compass direction."""
        if heading_deg is None:
            return "STATIONARY"

        deg = heading_deg % 360.0
        for name, lower, upper in self.COMPASS_SECTORS:
            if lower <= deg < upper:
                return name
        return "E"

    def compute_kinematics(
        self,
        observations: List[TrackObservation],
        fps: float,
    ) -> Tuple[List[TrajectorySegment], List[KinematicObservation]]:
        """
        Process a sequence of tracking observations for a video.

        Args:
            observations: List of all TrackObservation objects from the video run.
            fps: Actual video source frames per second.

        Returns:
            Tuple of:
                - List of TrajectorySegment metadata
                - List of KinematicObservation records
        """
        if not observations or fps <= 0:
            return [], []

        dt = 1.0 / fps

        # 1. Group observations by (object_type, track_id)
        grouped = defaultdict(list)
        for obs in observations:
            grouped[(obs.object_type, obs.track_id)].append(obs)

        all_segments: List[TrajectorySegment] = []
        all_kinematics: List[KinematicObservation] = []

        for (obj_type, tid), track_obs in grouped.items():
            # Sort chronologically by frame_id
            track_obs.sort(key=lambda o: o.frame_id)

            # 2. Partition into continuous segments where frame_gap <= max_continuous_gap
            current_seg_obs: List[TrackObservation] = [track_obs[0]]
            segment_list: List[List[TrackObservation]] = []

            for prev_o, curr_o in zip(track_obs[:-1], track_obs[1:]):
                gap = curr_o.frame_id - prev_o.frame_id
                if gap <= self.max_continuous_gap:
                    current_seg_obs.append(curr_o)
                else:
                    segment_list.append(current_seg_obs)
                    current_seg_obs = [curr_o]
            if current_seg_obs:
                segment_list.append(current_seg_obs)

            # 3. Analyze each continuous segment
            v_name = track_obs[0].video_name
            v_stem = v_name.rsplit(".", 1)[0] if "." in v_name else v_name

            for seg_idx, seg_obs in enumerate(segment_list, start=1):
                seg_id = f"{v_stem}_{obj_type}_{tid:03d}_seg_{seg_idx:03d}"
                n_obs = len(seg_obs)
                start_f = seg_obs[0].frame_id
                end_f = seg_obs[-1].frame_id
                duration_s = (end_f - start_f + 1) / fps
                is_valid = (n_obs >= self.min_segment_length)
                seg_class = "VALID_MOTION_SEGMENT" if is_valid else "INSUFFICIENT_TEMPORAL_DEPTH"

                # Path length & displacement
                path_len = 0.0
                for a, b in zip(seg_obs[:-1], seg_obs[1:]):
                    # For persons, anchor can be footpoint or centroid. Here we track center for motion.
                    path_len += math.hypot(b.center_x - a.center_x, b.center_y - a.center_y)
                displacement = math.hypot(seg_obs[-1].center_x - seg_obs[0].center_x,
                                          seg_obs[-1].center_y - seg_obs[0].center_y)

                segment_meta = TrajectorySegment(
                    segment_id=seg_id,
                    video_name=v_name,
                    object_type=obj_type,
                    track_id=tid,
                    class_name=seg_obs[0].class_name,
                    observations_count=n_obs,
                    start_frame=start_f,
                    end_frame=end_f,
                    duration_seconds=duration_s,
                    is_valid_motion=is_valid,
                    total_path_length_px=path_len,
                    displacement_px=displacement,
                )
                all_segments.append(segment_meta)

                # 4. Compute Velocity & Direction (N-1 steps)
                # And Acceleration (N-2 steps)
                seg_velocity_steps: List[Dict[str, Any]] = []

                prev_heading: Optional[float] = None

                for k in range(n_obs - 1):
                    o_from = seg_obs[k]
                    o_to = seg_obs[k + 1]

                    dx = o_to.center_x - o_from.center_x
                    dy = o_to.center_y - o_from.center_y
                    dist = math.hypot(dx, dy)

                    vx = dx * fps
                    vy = dy * fps
                    speed = dist * fps

                    is_stat = (speed < self.stationary_threshold)
                    if is_stat:
                        heading = None
                        d_8way = "STATIONARY"
                        turn_deg = 0.0 if prev_heading is not None else None
                    else:
                        # Image coordinates: +X right, +Y down
                        theta_rad = math.atan2(dy, dx)
                        heading = (math.degrees(theta_rad) + 360.0) % 360.0
                        d_8way = self.heading_to_8way(heading)

                        if prev_heading is not None:
                            # Turn angle in [-180, +180]
                            turn_deg = (heading - prev_heading + 180.0) % 360.0 - 180.0
                        else:
                            turn_deg = 0.0
                        prev_heading = heading

                    seg_velocity_steps.append({
                        "from_obs": o_from,
                        "to_obs": o_to,
                        "dx": dx,
                        "dy": dy,
                        "dist": dist,
                        "vx": vx,
                        "vy": vy,
                        "speed": speed,
                        "is_stationary": is_stat,
                        "heading": heading,
                        "d_8way": d_8way,
                        "turn_deg": turn_deg,
                    })

                # Compute acceleration for consecutive velocity steps (N-2 steps)
                for step_idx in range(len(seg_velocity_steps)):
                    curr_step = seg_velocity_steps[step_idx]

                    if step_idx > 0:
                        prev_step = seg_velocity_steps[step_idx - 1]
                        # dv / dt where dt = 1 / fps
                        ax = (curr_step["vx"] - prev_step["vx"]) * fps
                        ay = (curr_step["vy"] - prev_step["vy"]) * fps
                        acc = math.hypot(ax, ay)
                    else:
                        ax = None
                        ay = None
                        acc = None

                    kin_obs = KinematicObservation(
                        video_name=v_name,
                        object_type=obj_type,
                        track_id=tid,
                        segment_id=seg_id,
                        segment_classification=seg_class,
                        from_frame=curr_step["from_obs"].frame_id,
                        to_frame=curr_step["to_obs"].frame_id,
                        from_time_s=curr_step["from_obs"].timestamp_seconds,
                        to_time_s=curr_step["to_obs"].timestamp_seconds,
                        dt_s=dt,
                        from_x=curr_step["from_obs"].center_x,
                        from_y=curr_step["from_obs"].center_y,
                        to_x=curr_step["to_obs"].center_x,
                        to_y=curr_step["to_obs"].center_y,
                        dx_px=curr_step["dx"],
                        dy_px=curr_step["dy"],
                        step_distance_px=curr_step["dist"],
                        vx_px_s=curr_step["vx"],
                        vy_px_s=curr_step["vy"],
                        speed_px_s=curr_step["speed"],
                        is_stationary=curr_step["is_stationary"],
                        heading_deg=curr_step["heading"],
                        direction_8way=curr_step["d_8way"],
                        turn_angle_deg=curr_step["turn_deg"],
                        ax_px_s2=ax,
                        ay_px_s2=ay,
                        acceleration_px_s2=acc,
                    )
                    all_kinematics.append(kin_obs)

        return all_segments, all_kinematics

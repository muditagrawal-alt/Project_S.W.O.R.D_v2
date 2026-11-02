"""
Visualization module for S.W.O.R.D Phase 2 Generic Tracking Pipeline.
Renders annotated video frames with:
- Dual-stream bounding boxes (Weapon: Red/Orange, Person: Blue/Cyan)
- Separate Track IDs (#W{id}, #P{id})
- Estimated spatio-temporal association links
- Configured polygon zones
- Motion trajectory breadcrumb trails
And generates research-ready kinematic summary plots.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ..tracking.schema import TrackObservation
    from ..association.person_weapon_matcher import AssociationRecord
    from ..analytics.kinematics import KinematicObservation
except (ImportError, ValueError):
    from persistent_video_tracking.tracking.schema import TrackObservation
    from persistent_video_tracking.association.person_weapon_matcher import AssociationRecord
    from persistent_video_tracking.analytics.kinematics import KinematicObservation



class VideoVisualizer:
    """
    Renders high-quality security HUD annotations onto video frames.
    """

    COLOR_PERSON_BOX = (255, 180, 0)      # Cyan / Light Blue in BGR
    COLOR_PERSON_POINT = (255, 220, 50)
    COLOR_WEAPON_BOX = (0, 70, 255)       # Bright Red/Orange in BGR
    COLOR_WEAPON_POINT = (0, 140, 255)
    COLOR_ASSOCIATION = (0, 255, 255)     # Yellow line for estimated association
    COLOR_ZONE_BORDER = (50, 205, 50)     # Lime green
    COLOR_ZONE_FILL = (0, 100, 0)

    def __init__(
        self,
        zones_dict: Optional[Dict[str, np.ndarray]] = None,
        trail_length: int = 25,
    ):
        self.zones = zones_dict or {}
        self.trail_length = int(trail_length)
        # History for trails: (object_type, track_id) -> deque of (x, y)
        self.trails: Dict[Tuple[str, int], deque] = defaultdict(lambda: deque(maxlen=self.trail_length))

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_s: float,
        weapon_obs: List[TrackObservation],
        person_obs: List[TrackObservation],
        associations: List[AssociationRecord],
    ) -> np.ndarray:
        """Draw zones, tracks, associations, and HUD onto a single frame."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Zones (semi-transparent overlay)
        if self.zones:
            overlay = annotated.copy()
            for zone_name, pts in self.zones.items():
                pts_int = pts.astype(np.int32)
                cv2.polylines(annotated, [pts_int], isClosed=True, color=self.COLOR_ZONE_BORDER, thickness=2)
                cv2.fillPoly(overlay, [pts_int], color=self.COLOR_ZONE_FILL)
                # Label at first point
                cv2.putText(
                    annotated,
                    f"ZONE: {zone_name}",
                    (int(pts_int[0][0]) + 5, int(pts_int[0][1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 255, 220),
                    1,
                    cv2.LINE_AA,
                )
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        # 2. Update Trails
        for obs in person_obs:
            self.trails[("person", obs.track_id)].append((int(obs.bottom_center_x), int(obs.bottom_center_y)))
        for obs in weapon_obs:
            self.trails[("weapon", obs.track_id)].append((int(obs.center_x), int(obs.center_y)))

        # Draw Person Trails
        for obs in person_obs:
            pts = list(self.trails[("person", obs.track_id)])
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], self.COLOR_PERSON_POINT, 1, cv2.LINE_AA)

        # Draw Weapon Trails
        for obs in weapon_obs:
            pts = list(self.trails[("weapon", obs.track_id)])
            for i in range(1, len(pts)):
                cv2.line(annotated, pts[i - 1], pts[i], self.COLOR_WEAPON_POINT, 2, cv2.LINE_AA)

        # Map person observations by track_id for association drawing
        persons_by_id = {p.track_id: p for p in person_obs}

        # 3. Draw Person Bounding Boxes
        for p in person_obs:
            x1, y1, x2, y2 = int(p.x1), int(p.y1), int(p.x2), int(p.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_PERSON_BOX, 2)
            label = f"#P{p.track_id} person {p.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), self.COLOR_PERSON_BOX, -1)
            cv2.putText(annotated, label, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            # Footpoint dot
            cv2.circle(annotated, (int(p.bottom_center_x), int(p.bottom_center_y)), 4, self.COLOR_PERSON_POINT, -1)

        # 4. Draw Weapon Bounding Boxes & Association Links
        assoc_by_wid = {a.weapon_track_id: a for a in associations}

        for w_obs in weapon_obs:
            x1, y1, x2, y2 = int(w_obs.x1), int(w_obs.y1), int(w_obs.x2), int(w_obs.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_WEAPON_BOX, 2)

            # Association tag if present
            assoc = assoc_by_wid.get(w_obs.track_id)
            assoc_tag = ""
            if assoc and assoc.association_status == "ASSOCIATED" and assoc.person_track_id in persons_by_id:
                assoc_tag = f" -> #P{assoc.person_track_id}"
                # Draw association dashed/solid line from weapon center to person center
                p_matched = persons_by_id[assoc.person_track_id]
                cv2.line(
                    annotated,
                    (int(w_obs.center_x), int(w_obs.center_y)),
                    (int(p_matched.center_x), int(p_matched.center_y)),
                    self.COLOR_ASSOCIATION,
                    2,
                    cv2.LINE_AA,
                )

            label = f"#W{w_obs.track_id} {w_obs.class_name} {w_obs.confidence:.2f}{assoc_tag}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), self.COLOR_WEAPON_BOX, -1)
            cv2.putText(annotated, label, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # Centroid dot
            cv2.circle(annotated, (int(w_obs.center_x), int(w_obs.center_y)), 4, self.COLOR_WEAPON_POINT, -1)

        # 5. Header HUD
        hud_text = (
            f"Frame: {frame_id} | Time: {timestamp_s:.2f}s | "
            f"Weapons: {len(weapon_obs)} | Persons: {len(person_obs)} | "
            f"Associations: {sum(1 for a in associations if a.association_status == 'ASSOCIATED')}"
        )
        cv2.rectangle(annotated, (0, 0), (w, 28), (15, 15, 25), -1)
        cv2.putText(annotated, hud_text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

        return annotated

    @staticmethod
    def generate_kinematic_plots(
        kinematics: List[KinematicObservation],
        output_plot_path: str,
        video_name: str,
    ) -> None:
        """Generate velocity & acceleration timeline plots."""
        if not kinematics:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"S.W.O.R.D. Phase 2 Kinematics Timeline — {video_name}", fontsize=14, fontweight="bold")

        # Group by (object_type, track_id)
        grouped = defaultdict(list)
        for k in kinematics:
            grouped[(k.object_type, k.track_id)].append(k)

        for (obj_type, tid), items in grouped.items():
            times = [item.to_time_s for item in items]
            speeds = [item.speed_px_s for item in items]
            color = "tab:red" if obj_type == "weapon" else "tab:blue"
            prefix = "W" if obj_type == "weapon" else "P"
            ax1.plot(times, speeds, label=f"#{prefix}{tid} ({obj_type})", color=color, alpha=0.7, lw=1.2)

            acc_times = [item.to_time_s for item in items if item.acceleration_px_s2 is not None]
            acc_vals = [item.acceleration_px_s2 for item in items if item.acceleration_px_s2 is not None]
            if acc_times:
                ax2.plot(acc_times, acc_vals, label=f"#{prefix}{tid}", color=color, alpha=0.7, lw=1.2)

        ax1.set_ylabel("Speed (px/s)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=8, ncol=2)

        ax2.set_ylabel("Acceleration (px/s²)", fontsize=11)
        ax2.set_xlabel("Video Time (seconds)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150)
        plt.close(fig)

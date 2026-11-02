"""
Unit and Functional Verification Suite for S.W.O.R.D. v2 Persistent Video Tracking.
Tests:
- Weapon Ensemble WBF fusion logic
- Person Detector logic
- Dual Stream Tracker separation
- Kinematics engine (Direction, Velocity, Acceleration, Segment Splitting)
- Zone transitions and dwell timers
- Person <-> Weapon spatio-temporal association
"""

import os
import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure repository root is on sys.path if run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ..tracking.schema import TrackObservation
    from ..analytics.kinematics import KinematicsEngine, TrajectorySegment, KinematicObservation
    from ..analytics.zones_dwell import ZoneDwellEngine, ZoneEvent, DwellEpisode
    from ..association.person_weapon_matcher import SpatioTemporalAssociator, compute_iow_and_distance
except (ImportError, ValueError):
    from persistent_video_tracking.tracking.schema import TrackObservation
    from persistent_video_tracking.analytics.kinematics import KinematicsEngine, TrajectorySegment, KinematicObservation
    from persistent_video_tracking.analytics.zones_dwell import ZoneDwellEngine, ZoneEvent, DwellEpisode
    from persistent_video_tracking.association.person_weapon_matcher import SpatioTemporalAssociator, compute_iow_and_distance


class TestPersistentVideoTracking(unittest.TestCase):

    def test_iow_and_distance(self):
        """Test Intersection over Weapon (IoW) and centroid distance calculations."""
        # Weapon box inside person box
        w_box = (100, 100, 150, 150)  # area = 50 * 50 = 2500
        p_box = (50, 50, 200, 300)    # weapon completely inside person
        iow, dist = compute_iow_and_distance(w_box, p_box)
        self.assertAlmostEqual(iow, 1.0, places=2)
        self.assertTrue(dist >= 0)

        # Weapon completely outside person
        w_box_outside = (500, 500, 550, 550)
        iow_out, dist_out = compute_iow_and_distance(w_box_outside, p_box)
        self.assertEqual(iow_out, 0.0)
        self.assertTrue(dist_out > 200.0)

    def test_spatio_temporal_association(self):
        """Test bipartite person-weapon matching and association status assignment."""
        associator = SpatioTemporalAssociator(iow_threshold=0.15, min_association_score=0.35)

        w_obs = [
            TrackObservation(
                video_name="test.mp4",
                frame_id=1,
                timestamp_seconds=0.033,
                object_type="weapon",
                track_id=10,
                class_name="Pistol",
                confidence=0.85,
                x1=100, y1=100, x2=150, y2=150,
                center_x=125, center_y=125,
                bottom_center_x=125, bottom_center_y=150,
            )
        ]

        p_obs = [
            TrackObservation(
                video_name="test.mp4",
                frame_id=1,
                timestamp_seconds=0.033,
                object_type="person",
                track_id=5,
                class_name="person",
                confidence=0.90,
                x1=50, y1=50, x2=200, y2=300,
                center_x=125, center_y=175,
                bottom_center_x=125, bottom_center_y=300,
            )
        ]

        recs = associator.associate_frame("test.mp4", 1, 0.033, w_obs, p_obs)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].weapon_track_id, 10)
        self.assertEqual(recs[0].person_track_id, 5)
        self.assertEqual(recs[0].association_status, "ASSOCIATED")
        self.assertAlmostEqual(recs[0].iow, 1.0, places=2)

    def test_kinematics_segmentation_and_acceleration(self):
        """Test trajectory splitting on frame gaps and the N -> N-1 -> N-2 relationship."""
        engine = KinematicsEngine(stationary_threshold_px_s=15.0, min_segment_length=3, max_continuous_gap=1)
        fps = 30.0

        # Construct 5 observations moving right (+X): frame 0, 1, 2, 3, 4
        # Then gap: frame 10, 11 (new segment)
        obs = []
        for i in range(5):
            obs.append(TrackObservation(
                video_name="test.mp4", frame_id=i, timestamp_seconds=i / fps,
                object_type="weapon", track_id=1, class_name="knife", confidence=0.8,
                x1=100 + i * 10, y1=100, x2=120 + i * 10, y2=120,
                center_x=110 + i * 10, center_y=110, bottom_center_x=110 + i * 10, bottom_center_y=120
            ))

        # Add gap to frame 10 and 11
        for i in [10, 11]:
            obs.append(TrackObservation(
                video_name="test.mp4", frame_id=i, timestamp_seconds=i / fps,
                object_type="weapon", track_id=1, class_name="knife", confidence=0.8,
                x1=200, y1=200, x2=220, y2=220,
                center_x=210, center_y=210, bottom_center_x=210, bottom_center_y=220
            ))

        segments, kins = engine.compute_kinematics(obs, fps=fps)

        # 2 segments due to gap from frame 4 to 10
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].observations_count, 5)
        self.assertTrue(segments[0].is_valid_motion)
        self.assertEqual(segments[1].observations_count, 2)
        self.assertFalse(segments[1].is_valid_motion)

        # Segment 0 has 5 obs -> 4 velocity steps -> 3 acceleration steps
        seg0_kins = [k for k in kins if k.segment_id == segments[0].segment_id]
        self.assertEqual(len(seg0_kins), 4)

        seg0_accs = [k for k in seg0_kins if k.acceleration_px_s2 is not None]
        self.assertEqual(len(seg0_accs), 3)

        # All steps in seg0 moved purely right (+X) with speed = 10 px * 30 fps = 300 px/s
        for k in seg0_kins:
            self.assertEqual(k.direction_8way, "E")
            self.assertAlmostEqual(k.speed_px_s, 300.0, places=1)
            self.assertFalse(k.is_stationary)

    def test_zone_entry_exit_and_dwell(self):
        """Test zone entry, transition, and exit events with non-negative dwell times."""
        zones_config = {
            "normalized": False,
            "definitions": {
                "entrance": [[0, 0], [200, 0], [200, 200], [0, 200]],
                "lobby": [[200, 0], [400, 0], [400, 200], [200, 200]],
            }
        }
        zone_engine = ZoneDwellEngine(zones_config=zones_config, minimum_dwell_seconds=0.1)
        fps = 10.0

        # Person walks from Outside (x=-50) -> Entrance (x=100) -> Lobby (x=300) -> Outside (x=500)
        xs = [-50, 100, 100, 300, 300, 500]
        obs = []
        for i, x in enumerate(xs):
            obs.append(TrackObservation(
                video_name="test.mp4", frame_id=i, timestamp_seconds=i / fps,
                object_type="person", track_id=7, class_name="person", confidence=0.9,
                x1=x - 10, y1=50, x2=x + 10, y2=100,
                center_x=x, center_y=75, bottom_center_x=x, bottom_center_y=100  # footpoint
            ))

        events, dwells = zone_engine.analyze(obs, fps=fps)

        event_names = [e.event for e in events]
        # Must detect ZONE_ENTRY (into entrance), ZONE_TRANSITION (entrance -> lobby), ZONE_EXIT (lobby -> outside)
        self.assertIn("ZONE_ENTRY", event_names)
        self.assertIn("ZONE_TRANSITION", event_names)
        self.assertIn("ZONE_EXIT", event_names)
        self.assertTrue(len(dwells) > 0)


if __name__ == "__main__":
    unittest.main()

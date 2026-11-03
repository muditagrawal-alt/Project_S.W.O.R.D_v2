"""The team's persistent_video_tracking package (Phase 2) and its adapter to the Phase2Tracker interface."""

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from persistent_video_tracking.analytics.zones_dwell import ZoneDwellEngine  # noqa: E402
from persistent_video_tracking.tracking.dual_tracker import DualStreamTracker  # noqa: E402
from persistent_video_tracking.tracking.schema import TrackObservation  # noqa: E402

EMPTY = sv.Detections.empty()


def dets(boxes, scores):
    return sv.Detections(xyxy=np.asarray(boxes, np.float32).reshape(-1, 4), confidence=np.asarray(scores, np.float32),
                         class_id=np.zeros(len(scores), int))


def person_ids(tracker, d, frame_id):
    obs, _, _ = tracker.update(EMPTY, d, "v", frame_id, frame_id / 30.0)
    return [o.track_id for o in obs if o.object_type == "person"]


def test_tracks_age_on_frames_without_detections():
    """A track must expire after its lost buffer even when the intervening frames have no detections at all."""
    trk = DualStreamTracker(person_lost_buffer=5)
    box = [100, 100, 150, 250]
    first = person_ids(trk, dets([box], [0.9]), 0)
    for f in range(1, 21):                       # 20 empty frames, far beyond the 5-frame buffer
        person_ids(trk, EMPTY, f)
    person_ids(trk, dets([box], [0.9]), 21)      # a new track is reported from its second confirming frame
    again = person_ids(trk, dets([box], [0.9]), 22)
    assert first and again and again[0] != first[0]


def test_buffer_is_in_seconds_at_the_given_frame_rate():
    """lost_track_buffer counts frames at 30 fps; at 2 fps a 30-frame buffer must last 1 s (2 frames), not 15 s."""
    trk = DualStreamTracker(person_lost_buffer=30, frame_rate=2.0)
    box = [100, 100, 150, 250]
    first = person_ids(trk, dets([box], [0.9]), 0)
    for f in range(1, 5):                        # 2 s without the person
        person_ids(trk, EMPTY, f)
    person_ids(trk, dets([box], [0.9]), 5)
    assert person_ids(trk, dets([box], [0.9]), 6)[0] != first[0]


def test_low_score_detections_continue_a_track():
    """ByteTrack's second association: a detection between 0.1 and the activation threshold keeps an existing track."""
    trk = DualStreamTracker()
    box = [100, 100, 150, 250]
    first = person_ids(trk, dets([box], [0.9]), 0)
    assert person_ids(trk, dets([[102, 101, 152, 251]], [0.2]), 1) == first


def test_legacy_mode_reproduces_the_original_behaviour():
    """update_on_empty=False keeps the original behaviour (for benchmarking it as submitted)."""
    trk = DualStreamTracker(person_lost_buffer=5, update_on_empty=False)
    box = [100, 100, 150, 250]
    first = person_ids(trk, dets([box], [0.9]), 0)
    for f in range(1, 21):
        person_ids(trk, EMPTY, f)
    assert person_ids(trk, dets([box], [0.9]), 21) == first


def test_zone_dwell_counts_an_entry_at_frame_zero():
    eng = ZoneDwellEngine(zones_config={"definitions": {"door": [[0, 0], [200, 0], [200, 200], [0, 200]]}},
                          minimum_dwell_seconds=1.0)
    obs = [TrackObservation("v", f, f / 10.0, "person", 1, "person", 0.9, 50, 50, 70, 100 if f < 20 else 400,
                            60, 75, 60, 100 if f < 20 else 400) for f in range(25)]
    events, dwells = eng.analyze(obs, fps=10.0)
    exit_ = next(e for e in events if e.event == "ZONE_EXIT")
    assert exit_.dwell_seconds == pytest.approx(2.0)
    assert any(d.dwell_type == "ZONE_BASED" and d.duration_seconds == pytest.approx(2.0) for d in dwells)


class FakePersonDetector:
    """Returns scripted person detections (no model)."""

    def __init__(self, script):
        self.script, self.i = script, 0

    def detect(self, frame):
        boxes, scores = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return dets(boxes, scores) if boxes else EMPTY


def test_adapter_satisfies_the_phase2_contract():
    from tracking.contract import check_output
    from tracking.persistent_phase2 import PersistentPhase2

    script = [([[100 + 4 * i, 100, 150 + 4 * i, 250]], [0.9]) for i in range(10)]
    p2 = PersistentPhase2(fps=10.0, device="cpu", detector=FakePersonDetector(script))
    frame = np.zeros((480, 640, 3), np.uint8)
    ids = set()
    for i in range(10):
        people, events = p2.update(frame, i / 10.0)
        assert check_output(people, events, frame.shape, i / 10.0) == []
        ids |= {p["track_id"] for p in people}
    assert len(ids) == 1
    traj = people[0]["trajectory"]
    assert traj and traj["velocity_px_s"][0] > 0 and "speed_heights_s" in traj   # moving right
    segments, kinematics, zone_events, dwells = p2.finalize()
    assert segments and kinematics
    p2.reset()
    p2.detector = FakePersonDetector(script)
    assert p2.update(frame, 0.0)[0][0]["track_id"] == 1                            # fresh ID namespace

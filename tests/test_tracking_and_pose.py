"""Phase 2 (tracking, trajectory, association, schema) and Phase 3 (track-keyed pose velocity) tests."""

import numpy as np
import pytest

from tracking.association import associate
from tracking.schema import SchemaError, empty_pose, validate
from tracking.tracker import ByteTrackWrapper
from tracking.trajectory import TrajectoryAnalyzer, apply_homography


# ───────────────────────────── ByteTrack wrapper ─────────────────────────────
def test_ids_survive_detection_order_swaps():
    """Two people walking apart; the detector lists them in a different order every frame."""
    trk = ByteTrackWrapper()
    ids_a, ids_b = set(), set()
    for f in range(30):
        a = [100 + 3 * f, 100, 160 + 3 * f, 300]
        b = [500 - 3 * f, 120, 560 - 3 * f, 320]
        boxes = np.array([a, b] if f % 2 == 0 else [b, a], dtype=np.float32)
        for t in trk.update(boxes, np.array([0.9, 0.9])):
            (ids_a if abs(t["bbox"][1] - 100) < 5 else ids_b).add(t["track_id"])  # A is the box at y1=100
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_track_survives_short_occlusion():
    trk = ByteTrackWrapper(track_buffer=30)
    seen = []
    for f in range(40):
        box = np.array([[100 + 2 * f, 100, 160 + 2 * f, 300]], dtype=np.float32)
        out = trk.update(box, np.array([0.9])) if not (15 <= f < 20) else trk.update(np.zeros((0, 4)), np.zeros(0))
        seen += [t["track_id"] for t in out]
    assert len(set(seen)) == 1


# ───────────────────────────── trajectory ─────────────────────────────
def test_constant_velocity_and_heading():
    ta = TrajectoryAnalyzer(window_s=0.5)
    fps = 25.0
    for f in range(50):
        t = f / fps
        x = 100 + 50 * t  # 50 px/s to the right
        feats, _ = ta.update(t, [{"track_id": 1, "bbox": [x - 20, 100, x + 20, 300]}])
    f1 = feats[1]
    assert f1["velocity_px_s"][0] == pytest.approx(50, abs=0.5)
    assert abs(f1["velocity_px_s"][1]) < 0.5
    assert f1["heading_deg"] == pytest.approx(0, abs=1)
    assert f1["speed_heights_s"] == pytest.approx(50 / 200, abs=0.01)
    assert abs(f1["acceleration_px_s2"][0]) < 1.0
    assert f1["dwell_s"] == pytest.approx(49 / fps)


def test_acceleration_estimate():
    ta = TrajectoryAnalyzer(window_s=0.5)
    for f in range(50):
        t = f / 25.0
        x = 100 + 0.5 * 80 * t * t  # a = 80 px/s²
        feats, _ = ta.update(t, [{"track_id": 1, "bbox": [x - 20, 100, x + 20, 300]}])
    assert feats[1]["acceleration_px_s2"][0] == pytest.approx(80, rel=0.05)


def test_zone_enter_exit_events():
    ta = TrajectoryAnalyzer(zones=[{"name": "door", "polygon": [[200, 0], [300, 0], [300, 400], [200, 400]]}])
    events = []
    for f in range(60):
        x = 100 + 5 * f
        _, ev = ta.update(f / 25.0, [{"track_id": 7, "bbox": [x - 10, 100, x + 10, 300]}])
        events += ev
    kinds = [(e["type"], e["zone"]) for e in events]
    assert kinds == [("zone_enter", "door"), ("zone_exit", "door")]


def test_homography_gives_metres():
    H = np.diag([0.01, 0.01, 1.0])  # 100 px = 1 m
    assert apply_homography(H, 250, 400) == pytest.approx((2.5, 4.0))
    ta = TrajectoryAnalyzer(homography=H)
    for f in range(30):
        t = f / 25.0
        x = 100 + 150 * t  # 150 px/s = 1.5 m/s
        feats, _ = ta.update(t, [{"track_id": 1, "bbox": [x - 20, 100, x + 20, 300]}])
    assert feats[1]["speed_m_s"] == pytest.approx(1.5, abs=0.02)


# ───────────────────────────── association ─────────────────────────────
def _person(tid, box, wrist_l=None, wrist_r=None):
    kp, sc = empty_pose()
    if wrist_l:
        kp[9], sc[9] = list(wrist_l), 0.9
    if wrist_r:
        kp[10], sc[10] = list(wrist_r), 0.9
    return {"track_id": tid, "bbox": box, "keypoints": kp, "scores": sc}


def test_weapon_goes_to_the_nearest_wrist():
    persons = [_person(1, [0, 0, 100, 300], wrist_r=(90, 150)), _person(2, [200, 0, 300, 300], wrist_l=(210, 150))]
    out = associate([{"bbox": [95, 140, 130, 170]}], persons)
    assert out[0]["person_id"] == 1 and out[0]["hand"] == "right" and out[0]["method"] == "wrist"


def test_far_weapon_is_unattended():
    persons = [_person(1, [0, 0, 100, 300], wrist_r=(90, 150))]
    assert associate([{"bbox": [600, 600, 650, 640]}], persons)[0]["person_id"] is None


def test_box_fallback_without_pose():
    persons = [{"track_id": 4, "bbox": [0, 0, 100, 300]}]
    out = associate([{"bbox": [20, 100, 60, 140]}], persons)
    assert out[0]["person_id"] == 4 and out[0]["method"] == "box"


# ───────────────────────────── schema ─────────────────────────────
def test_schema_validation():
    kp, sc = empty_pose()
    doc = {"video": "v.mp4", "fps": 25.0, "width": 640, "height": 480,
           "frames": [{"frame_idx": 0, "persons": [{"track_id": 1, "bbox": [0, 0, 1, 1], "keypoints": kp, "scores": sc}],
                       "weapons": [{"track_id": 2, "cls": 0, "bbox": [0, 0, 1, 1]}]}]}
    validate(doc)
    doc["frames"][0]["persons"][0]["keypoints"] = kp[:5]
    with pytest.raises(SchemaError):
        validate(doc)


# ───────────────────────────── Phase 3: velocity keyed by track ─────────────────────────────
def _bare_pose_pipeline():
    """RTMPosePipeline without loading ONNX models (only the velocity logic is exercised)."""
    from pose_pipeline import RTMPosePipeline

    p = RTMPosePipeline.__new__(RTMPosePipeline)
    p.kpt_thr, p.velocity_ema_alpha, p.max_gap_s, p.stale_s = 0.35, 1.0, 0.5, 2.0
    p.track_state = {}
    return p


def _kpts(wrist_x):
    k = np.zeros((17, 2), dtype=np.float32)
    s = np.full(17, 0.9, dtype=np.float32)
    k[9] = (wrist_x, 200)
    k[10] = (wrist_x + 40, 200)
    return k, s


def test_velocity_follows_track_not_detection_order():
    """Person A at x≈100 and B at x≈500, both still. The detector swaps their order between frames.
    Keyed by detection index (old behaviour) this looks like a 400 px jump; keyed by track ID it is ~0."""
    p = _bare_pose_pipeline()
    a, b = _kpts(100), _kpts(500)
    p.compute_wrist_velocity(11, *a, current_time=0.00)
    p.compute_wrist_velocity(12, *b, current_time=0.00)
    va = p.compute_wrist_velocity(11, *a, current_time=0.04)  # same people, opposite order this frame
    vb = p.compute_wrist_velocity(12, *b, current_time=0.04)
    assert va["left_wrist_speed_px_s"] == 0.0 and vb["left_wrist_speed_px_s"] == 0.0

    old_style = _bare_pose_pipeline()  # index-keyed: index 0 was A, now is B
    old_style.compute_wrist_velocity(0, *a, current_time=0.00)
    swapped = old_style.compute_wrist_velocity(0, *b, current_time=0.04)
    assert swapped["left_wrist_speed_px_s"] == pytest.approx(400 / 0.04)


def test_velocity_not_reported_without_persistent_ids():
    p = _bare_pose_pipeline()
    v = p.compute_wrist_velocity(0, *_kpts(100), current_time=0.0, id_persistent=False)
    assert v["left_wrist_speed_px_s"] is None and v["id_persistent"] is False


def test_long_gap_restarts_estimate():
    p = _bare_pose_pipeline()
    p.compute_wrist_velocity(1, *_kpts(100), current_time=0.0)
    v = p.compute_wrist_velocity(1, *_kpts(300), current_time=2.0)  # 2 s later: not a velocity sample
    assert v["left_wrist_speed_px_s"] is None
    v = p.compute_wrist_velocity(1, *_kpts(310), current_time=2.1)
    assert v["left_wrist_speed_px_s"] == pytest.approx(100.0)


def test_reach_indicator_has_no_pixel_fallback():
    from pose_pipeline import analyze_hand_to_waist_distance

    k = np.zeros((17, 2), dtype=np.float32)
    s = np.zeros(17, dtype=np.float32)
    k[11], k[12], s[11], s[12] = (100, 200), (120, 200), 0.9, 0.9   # hips
    k[9], s[9] = (110, 230), 0.9                                    # wrist 30 px from the waist
    res = analyze_hand_to_waist_distance(k, s, 0.3)                 # shoulders unknown -> no torso length
    assert res["left_wrist_dist_px"] == pytest.approx(30.0)
    assert res["hands_near_waist"] is False                        # old code: True via the 55 px fallback


def test_weapons_are_not_dropped_by_track_confirmation():
    """Regression (2-fps CCTV): a hand-held weapon jumps between frames, so IoU tracking never confirms it.
    Every detection must still reach Phase 5, linked to its person; track_id is None while unconfirmed."""
    from tracking.pipeline import weapons_from_detections
    from tracking.tracker import ByteTrackWrapper

    trk = ByteTrackWrapper(track_buffer=5)
    kp = [[0.0, 0.0]] * 17
    seen = 0
    for i in range(6):
        x = 100 + 60 * i  # moves farther than its own width every frame -> IoU 0 with the previous box
        person = {"track_id": 1, "bbox": [x - 40, 50, x + 40, 300], "keypoints": [list(p) for p in kp], "scores": [0.9] * 17}
        person["keypoints"][10] = [x + 5.0, 160.0]  # right wrist inside the weapon box
        raw, weapons = weapons_from_detections(trk, [[x, 150, x + 37, 205]], [0.8], [0], [person])
        assert len(raw) == len(weapons) == 1
        assert weapons[0]["person_id"] == 1 and weapons[0]["assoc_method"] == "wrist"
        seen += 1
    assert seen == 6

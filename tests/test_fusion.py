"""Phase 5: evidence features, the temporal rules engine, the baseline ladder / metrics, and the transformer."""

import numpy as np
import pytest
import torch

from fusion.evaluate import b0_frame, baseline_ladder, evaluate_alarms, ladder_details, segments, verified_metrics
from fusion.features import FeatureConfig, person_features, weapon_plausibility
from fusion.rules import FusionConfig, TemporalThreatEngine
from fusion.transformer import FEATURES, TemporalThreatTransformer, train

SHAPE = (576, 768)
FPS = 10


def person(tid=1, bbox=(300, 200, 380, 420)):
    kp = [[340.0, 260.0]] * 17
    kp = kp[:10] + [[385.0, 310.0]] + kp[11:]   # right wrist on the default weapon box
    return {"track_id": tid, "bbox": list(bbox), "score": 0.9, "keypoints": kp, "scores": [0.9] * 17,
            "trajectory": {"speed_heights_s": 0.3, "zone": None}, "kinematics": None}


def weapon(score=0.8, bbox=(370, 300, 400, 320), pid=1, method="wrist"):
    return {"track_id": 7, "bbox": list(bbox), "cls": 0, "cls_name": "Pistol", "score": score, "person_id": pid,
            "hand": "right" if method == "wrist" else None, "assoc_dist_norm": 0.1, "assoc_method": method}


def frame(t, persons, weapons, raw=None):
    return {"frame_idx": int(round(t * FPS)), "timestamp": t, "persons": persons, "weapons": weapons,
            "weapon_detections": raw if raw is not None else [{"bbox": w["bbox"], "cls": 0, "score": w["score"]} for w in weapons],
            "actions": {}}


def run_engine(frames, cfg=None, actions=None):
    eng = TemporalThreatEngine(cfg)
    levels = []
    for f in frames:
        state, _ = eng.update(f["timestamp"], f, SHAPE, actions or {})
        levels.append(state["scene_level"])
    return levels


# ───────────────────────────── features ─────────────────────────────
def test_plausibility_discounts_frame_filling_boxes():
    cfg = FeatureConfig()
    area = SHAPE[0] * SHAPE[1]
    assert weapon_plausibility(weapon(), area, person(), cfg) == 1.0
    road = weapon(bbox=(0, 100, 768, 576))  # vtest-style: a curved road detected as a "Sword"
    assert weapon_plausibility(road, area, None, cfg) == 0.0


def test_person_features_hand_link_and_stale_action():
    cfg = FeatureConfig()
    f = person_features(person(), [weapon()], SHAPE, 5.0, None, cfg)
    assert f["weapon_hand"] == pytest.approx(0.8) and f["weapon_body"] == 0.0
    f = person_features(person(), [weapon(method="containment")], SHAPE, 5.0, None, cfg)
    assert f["weapon_body"] == pytest.approx(0.8) and f["weapon_hand"] == 0.0
    act = {"t": 1.0, "group_probs": {"prob_WEAPON_THREAT": 0.9}}
    assert person_features(person(), [], SHAPE, 2.0, act, cfg)["act_weapon"] == 0.9
    assert person_features(person(), [], SHAPE, 5.0, act, cfg)["act_weapon"] == 0.0  # older than action_max_age_s


# ───────────────────────────── rules engine ─────────────────────────────
def test_persistent_hand_held_weapon_escalates_then_holds_and_decays():
    ts = np.arange(0, 8, 1 / FPS)
    frames = [frame(t, [person()], [weapon()] if 1.0 <= t < 4.0 else []) for t in ts]
    lv = run_engine(frames)
    first_wre = ts[lv.index("WEAPON_RELATED_EVENT")]
    assert 2.0 <= first_wre <= 2.5, first_wre  # needs persist_s = 1 s of continuous evidence
    assert all(x == "NORMAL" for x, t in zip(lv, ts) if t < 1.0)
    # hysteresis: stays raised for hold_s after the evidence stops, then steps down
    assert lv[int(4.5 * FPS)] == "WEAPON_RELATED_EVENT"
    assert lv[-1] == "NORMAL"


def test_vanished_track_holds_scene_level_for_hold_s():
    ts = np.arange(0, 8, 1 / FPS)
    # armed person tracked until t=4, then lost (occlusion): scene must not drop the moment the track disappears
    frames = [frame(t, [person()] if t < 4.0 else [], [weapon()] if 1.0 <= t < 4.0 else []) for t in ts]
    lv = run_engine(frames)
    assert lv[int(3.9 * FPS)] == "WEAPON_RELATED_EVENT"
    assert lv[int(5.5 * FPS)] == "WEAPON_RELATED_EVENT"   # within hold_s = 2 s of last sighting
    assert lv[int(6.5 * FPS)] == "NORMAL"


def test_single_frame_blip_never_reaches_weapon_event():
    ts = np.arange(0, 5, 1 / FPS)
    frames = [frame(t, [person()], [weapon(score=0.95)] if abs(t - 2.0) < 1e-6 else []) for t in ts]
    assert "WEAPON_RELATED_EVENT" not in run_engine(frames)


def test_implausible_frame_filling_weapon_is_ignored():
    ts = np.arange(0, 5, 1 / FPS)
    road = weapon(score=0.9, bbox=(0, 100, 768, 576), pid=None, method=None)
    frames = [frame(t, [person()], [road]) for t in ts]
    assert set(run_engine(frames)) == {"NORMAL"}


def test_ablation_without_weapons_stays_normal():
    ts = np.arange(0, 5, 1 / FPS)
    frames = [frame(t, [person()], [weapon()]) for t in ts]
    assert set(run_engine(frames, FusionConfig(use_weapon=False))) == {"NORMAL"}
    assert "WEAPON_RELATED_EVENT" in run_engine(frames, FusionConfig())


def test_action_evidence_alone_reaches_suspicious_but_not_weapon_event():
    ts = np.arange(0, 6, 1 / FPS)
    frames = [frame(t, [person()], []) for t in ts]
    eng = TemporalThreatEngine()
    levels = []
    for f in frames:
        act = {1: {"t": f["timestamp"], "group_probs": {"prob_WEAPON_THREAT": 0.9}}}
        levels.append(eng.update(f["timestamp"], f, SHAPE, act)[0]["scene_level"])
    assert "SUSPICIOUS_ACTIVITY" in levels and "WEAPON_RELATED_EVENT" not in levels
    assert levels[0] == "NORMAL"  # a new track starts from zero evidence


def test_unattended_weapon_raises_scene_to_elevated():
    ts = np.arange(0, 5, 1 / FPS)
    frames = [frame(t, [], [weapon(score=0.9, pid=None, method=None)]) for t in ts]
    lv = run_engine(frames)
    assert lv[0] == "NORMAL" and lv[-1] == "ELEVATED"


# ───────────────────────────── metrics & ladder ─────────────────────────────
def test_segments_and_event_metrics():
    t = np.arange(0, 60, 0.5)
    gt = (t >= 20) & (t <= 25)
    alarm = ((t >= 19.5) & (t <= 26)) | ((t >= 40) & (t <= 41))
    gt_segs = segments(t, gt)
    assert gt_segs == [(20.0, 25.0)]
    m = evaluate_alarms(t, alarm, gt_segs, gt)
    assert m["event_recall"] == 1.0 and m["detection_delay_s_median"] == -0.5
    assert m["false_alarm_segments"] == 1 and m["alarm_segments"] == 2
    assert m["frame_recall"] == 1.0 and 0 < m["frame_precision"] < 1


def test_b0_uses_raw_detections_with_fallback():
    f_raw = frame(0.0, [], [], raw=[{"bbox": [0, 0, 5, 5], "cls": 0, "score": 0.7}])
    assert b0_frame([f_raw]) == [True]  # raw detection although no weapon track exists yet
    legacy = frame(0.0, [], [weapon(score=0.7)])
    del legacy["weapon_detections"]
    assert b0_frame([legacy]) == [True]


def test_baseline_ladder_keys_and_monotonic_filtering():
    ts = np.arange(0, 8, 1 / FPS)
    frames = [frame(t, [person()], [weapon()] if 1.0 <= t < 4.0 else []) for t in ts]
    lad = baseline_ladder(frames, SHAPE)
    assert set(lad) == {"B0_frame", "B0+_persist", "B1_track", "B1+_plausible", "B2_hand", "B3_rules_noaction",
                        "B4_rules_full", "B4_rules_full_WRE_only"}
    assert all(len(v) == len(frames) for v in lad.values())
    assert sum(lad["B1_track"]) <= sum(lad["B0_frame"])  # persistence only removes frames
    assert any(lad["B4_rules_full"])


# ───────────────────────────── transformer ─────────────────────────────
def test_transformer_forward_and_padding():
    m = TemporalThreatTransformer().eval()
    x = torch.randn(3, 50, len(FEATURES))
    assert m(x).shape == (3, 4)
    mask = torch.zeros(3, 50, dtype=torch.bool)
    mask[:, 40:] = True
    out_masked = m(x, mask)
    x2 = x.clone()
    x2[:, 40:] = 99.0  # padded steps must not matter
    assert torch.allclose(out_masked, m(x2, mask), atol=1e-5)


def test_transformer_learns_separable_toy_task():
    rng = np.random.default_rng(0)
    items = []
    for i in range(160):
        y = i % 2 * 3  # NORMAL vs WEAPON_RELATED_EVENT
        x = rng.normal(0, 0.1, (50, len(FEATURES))).astype(np.float32)
        x[:, 0] += 0.8 if y else 0.0
        items.append((x, y))
    model, hist = train(items, epochs=15, lr=2e-3)
    assert hist[-1] < 0.5 * hist[0]
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(np.stack([x for x, _ in items]))).argmax(1).numpy()
    assert (pred == np.array([y for _, y in items])).mean() > 0.95


def test_verified_metrics_reject_alarms_pointing_elsewhere():
    ts = np.arange(0, 8, 1 / FPS)
    gt_box = [370, 300, 400, 320]                   # weapon in the right hand of person 1
    in_gt = [(2.0 <= t < 5.0) for t in ts]
    gt_boxes = [[gt_box] if g else [] for g in in_gt]
    gt_segs = segments(ts, in_gt)
    wrong = weapon(score=0.9, bbox=(10, 10, 300, 200), pid=None, method=None)   # big box elsewhere, same time
    frames_wrong = [frame(t, [person()], [wrong] if g else []) for t, g in zip(ts, in_gt)]
    frames_right = [frame(t, [person()], [weapon(score=0.9)] if g else []) for t, g in zip(ts, in_gt)]
    for frames_, expect in ((frames_wrong, 0), (frames_right, 1)):
        d = ladder_details(frames_, SHAPE)
        v = verified_metrics(frames_, d, gt_boxes, gt_segs)
        temporal = evaluate_alarms(ts, d["B0_frame"]["flags"], gt_segs)
        assert temporal["event_recall"] == 1.0          # time-only metric credits both
        assert v["B0_frame"]["verified_events_detected"] == expect
        assert v["B0_frame"]["misattributed_alarm_segments"] == 1 - expect
    v = verified_metrics(frames_right, ladder_details(frames_right, SHAPE), gt_boxes, gt_segs)
    assert v["B4_rules_full"]["verified_events_detected"] == 1   # alarmed person 1 carries the weapon


def test_close_events_get_no_early_credit_from_the_previous_event():
    """Two weapon events 3 s apart and one continuous alarm: the second event must not be credited as warned
    early (its look-back window overlaps the first event), and the delay must not be negative."""
    t = np.arange(0, 30, 0.5)
    gt = ((t >= 10) & (t <= 12)) | ((t >= 15) & (t <= 17))
    alarm = (t >= 10) & (t <= 17.5)
    segs = segments(t, gt, merge_gap_s=1.0)
    assert len(segs) == 2
    m = evaluate_alarms(t, alarm, segs, gt, merge_gap_s=1.0)
    assert m["event_recall"] == 1.0 and m["delays_s"] == [0.0, 0.0]
    assert m["isolated_events"] == 1 and m["isolated_delays_s"] == [0.0]
    early = (t >= 7) & (t <= 17.5)   # genuine warning 3 s before an isolated event
    m2 = evaluate_alarms(t, early, segs, gt, merge_gap_s=1.0)
    assert m2["delays_s"][0] == -3.0 and m2["delays_s"][1] == 0.0

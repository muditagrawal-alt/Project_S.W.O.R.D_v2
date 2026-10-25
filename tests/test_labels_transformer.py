"""Interval labels (fusion/labels.py) and the scene-level Temporal Transformer path."""

import numpy as np
import pytest

from fusion.labels import Interval, load_labels, merge_adjacent, save_labels, scene_levels
from fusion.transformer import (SCENE_FEATURES, export_scene_sequences, load_model, predict_steps, save_model,
                                scene_windows, train, windows)


def test_label_roundtrip_and_validation(tmp_path):
    ivs = [Interval("a.mp4", 1.0, 3.0, "WEAPON_RELATED_EVENT", source="derived:boxes", notes="PAH1"),
           Interval("a.mp4", 2.0, 5.0, "SUSPICIOUS_ACTIVITY", scope="person", anchor={"t": 2.0, "bbox": [1, 2, 3, 4]}),
           Interval("b.mp4", 0.0, 9.0, "NORMAL")]
    save_labels(ivs, tmp_path / "l.csv")
    back = load_labels(tmp_path / "l.csv")
    assert back == ivs
    with pytest.raises(ValueError):
        Interval("a", 0, 1, "DANGER")
    with pytest.raises(ValueError):
        Interval("a", 0, 1, "NORMAL", scope="person")        # person scope needs an anchor
    with pytest.raises(ValueError):
        Interval("a", 2, 1, "NORMAL")


def test_scene_levels_highest_wins_and_unknown():
    ivs = [Interval("a", 1.0, 3.0, "SUSPICIOUS_ACTIVITY"), Interval("a", 2.0, 2.5, "WEAPON_RELATED_EVENT")]
    t = [0.0, 1.0, 2.2, 2.9, 4.0]
    assert scene_levels(ivs, t).tolist() == [0, 2, 3, 2, 0]
    assert scene_levels(ivs, t, exhaustive=False).tolist() == [-1, 2, 3, 2, -1]


def test_merge_adjacent():
    m = merge_adjacent([Interval("a", 0, 1, "NORMAL"), Interval("a", 1.2, 2, "NORMAL"), Interval("a", 5, 6, "NORMAL")], gap_s=0.5)
    assert [(i.start_s, i.end_s) for i in m] == [(0, 2), (5, 6)]


def _doc(n=60, armed_from=30):
    frames = []
    for i in range(n):
        t = i / 10
        p = {"track_id": 1, "bbox": [300, 200, 380, 420], "score": 0.9, "keypoints": [[340, 260]] * 10 + [[385, 310]] + [[340, 260]] * 6,
             "scores": [0.9] * 17, "trajectory": {"speed_heights_s": 0.2}, "kinematics": None}
        w = [{"track_id": None, "bbox": [370, 300, 400, 320], "cls": 0, "cls_name": "Pistol", "score": 0.9, "person_id": 1,
              "hand": "right", "assoc_dist_norm": 0.1, "assoc_method": "wrist"}] if i >= armed_from else []
        frames.append({"frame_idx": i, "timestamp": t, "persons": [p], "weapons": w,
                       "weapon_detections": [{"bbox": x["bbox"], "cls": 0, "score": x["score"]} for x in w], "actions": {}})
    return {"video": "v.mp4", "fps": 10.0, "width": 768, "height": 576, "frames": frames}


def test_scene_sequence_features():
    ts, x = export_scene_sequences(_doc(), rate_hz=5)
    assert x.shape == (len(ts), len(SCENE_FEATURES))
    hand = SCENE_FEATURES.index("max_weapon_hand")
    assert x[ts < 2.9, hand].max() == 0 and x[ts > 3.1, hand].min() == pytest.approx(0.9)
    assert np.allclose(x[:, SCENE_FEATURES.index("log1p_persons")], np.log1p(1))


def test_windows_padding_and_scene_windows():
    x = np.arange(10, dtype=np.float32).reshape(10, 1)
    seg, mask = windows(x, 2, 5)
    assert seg[:, 0].tolist() == [0, 0, 0, 1, 2] and mask.tolist() == [True, True, False, False, False]
    lv = np.array([0, 0, -1, 3, 3, 3, -1, 0, 0, 0])
    items = scene_windows(x, lv, win=5, stride=1)
    assert len(items) == 8 and {it[2] for it in items} == {0, 3}


def test_scene_transformer_learns_and_roundtrips(tmp_path):
    items = []
    for armed in (True, False) * 20:
        ts, x = export_scene_sequences(_doc(armed_from=30 if armed else 10**6), rate_hz=5)
        lv = np.where(ts >= 3.0, 3 if armed else 0, 0)
        items += scene_windows(x, lv, win=10, stride=1)
    model, hist = train(items, epochs=8, lr=3e-3)
    assert hist[-1] < hist[0]
    save_model(model, tmp_path / "m.pt", SCENE_FEATURES, {"win": 10})
    m2, ck = load_model(tmp_path / "m.pt")
    ts, x = export_scene_sequences(_doc(), rate_hz=5)
    p = predict_steps(m2, x, win=10)
    assert p.shape == (len(ts), 4) and p[ts > 4.5].argmax(1).tolist().count(3) >= 0.8 * (ts > 4.5).sum()
    assert ck["meta"]["win"] == 10

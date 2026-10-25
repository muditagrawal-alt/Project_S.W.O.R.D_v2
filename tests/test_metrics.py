"""Tests for ensemble/metrics.py — matching, operating point, COCO cross-check, truncation behaviour."""

import numpy as np
import pytest

from ensemble.metrics import (bootstrap_ci, coco_metrics, detection_metrics, load_yolo_labels, match_image,
                              operating_point)

NAMES = ["a", "b"]


def rec(image_id, gt, preds, w=640, h=480):
    gt = np.asarray(gt, dtype=float).reshape(-1, 5)
    preds = np.asarray(preds, dtype=float).reshape(-1, 6)
    return {"image_id": image_id, "path": f"{image_id}.jpg", "width": w, "height": h,
            "gt_cls": gt[:, 0].astype(int), "gt_boxes": gt[:, 1:5],
            "pred_cls": preds[:, 0].astype(int), "pred_scores": preds[:, 1], "pred_boxes": preds[:, 2:6]}


def test_load_yolo_labels(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1 0.5 0.5 0.25 0.5\n")
    cls, boxes = load_yolo_labels(str(p), 640, 480)
    assert cls.tolist() == [1]
    np.testing.assert_allclose(boxes, [[240, 120, 400, 360]])
    (tmp_path / "e.txt").write_text("")
    assert len(load_yolo_labels(str(tmp_path / "e.txt"), 640, 480)[0]) == 0


def test_duplicate_prediction_is_one_tp_one_fp():
    r = rec(0, [[0, 10, 10, 110, 110]], [[0, 0.9, 10, 10, 110, 110], [0, 0.8, 12, 12, 112, 112]])
    tp = match_image(r)[:, 0]
    assert tp.tolist() == [True, False]


def test_wrong_class_is_not_a_match():
    r = rec(0, [[0, 10, 10, 110, 110]], [[1, 0.9, 10, 10, 110, 110]])
    assert not match_image(r).any()


def test_operating_point_counts():
    recs = [rec(0, [[0, 10, 10, 110, 110]], [[0, 0.9, 10, 10, 110, 110], [0, 0.3, 300, 300, 350, 350]]),
            rec(1, [], [[1, 0.6, 50, 50, 90, 90]])]
    op = operating_point(recs, 0.5, n_classes=2)
    assert op["false_positives"] == 1          # only the 0.6 box on the empty image survives as an FP
    assert op["micro_recall"] == 1.0
    assert op["fp_per_image_without_gt"] == 1.0


def _synthetic(seed=0, n=60):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        gts, preds = [], []
        for _ in range(rng.integers(1, 4)):
            c = int(rng.integers(2))
            x, y = rng.uniform(0, 400), rng.uniform(0, 300)
            s = rng.uniform(40, 200)
            gts.append([c, x, y, x + s, y + s])
            if rng.uniform() < 0.85:  # detected, with jitter
                j = rng.normal(0, s * 0.05, 4)
                preds.append([c, rng.uniform(0.3, 1.0), x + j[0], y + j[1], x + s + j[2], y + s + j[3]])
        for _ in range(rng.integers(0, 3)):  # false positives
            x, y = rng.uniform(0, 400), rng.uniform(0, 300)
            preds.append([int(rng.integers(2)), rng.uniform(0.0, 0.6), x, y, x + 60, y + 60])
        recs.append(rec(i, gts, preds))
    return recs


def test_matches_independent_coco_evaluator():
    recs = _synthetic()
    ours = detection_metrics(recs, NAMES)
    coco = coco_metrics(recs, NAMES)
    assert abs(ours["mAP50-95"] - coco["AP"]) < 0.03
    assert abs(ours["mAP50"] - coco["AP50"]) < 0.03


def test_truncating_predictions_cannot_raise_ap():
    """Computing AP after a confidence cut (what the legacy scripts did) can only lower AP with current Ultralytics."""
    recs = _synthetic(seed=3)
    full = detection_metrics(recs, NAMES)["mAP50"]
    for thr in (0.1, 0.2, 0.4):
        cut = [{**r, "pred_cls": r["pred_cls"][r["pred_scores"] >= thr], "pred_boxes": r["pred_boxes"][r["pred_scores"] >= thr],
                "pred_scores": r["pred_scores"][r["pred_scores"] >= thr]} for r in recs]
        assert detection_metrics(cut, NAMES)["mAP50"] <= full + 1e-9


def test_coco_size_range_without_objects_is_none():
    recs = [rec(0, [[0, 0, 0, 300, 300]], [[0, 0.9, 0, 0, 300, 300]])]  # one large object only
    out = coco_metrics(recs, NAMES)
    assert out["AP_small"] is None and out["n_gt_small"] == 0
    assert out["AP_large"] == pytest.approx(1.0)


def test_paired_bootstrap_of_identical_predictions_is_zero():
    recs = _synthetic(seed=5, n=30)
    ci = bootstrap_ci(recs, lambda rs: detection_metrics(rs, NAMES)["mAP50"], n_boot=50, paired_with=recs)
    assert ci["point"] == 0 and ci["low"] == 0 and ci["high"] == 0

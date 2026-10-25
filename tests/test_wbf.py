"""WBF must reproduce the reference implementation (ensemble-boxes, by the WBF paper's authors)."""

import numpy as np
import pytest

ensemble_boxes = pytest.importorskip("ensemble_boxes")

from ensemble.wbf import weighted_boxes_fusion


def random_predictions(rng, n_models=3, n_objects=6, n_classes=3, boxes_per_object=(0, 3)):
    """Several models predicting jittered copies of the same objects; a model may emit 0–2 boxes per object."""
    objects = []
    for _ in range(n_objects):
        x1, y1 = rng.uniform(0, 0.7, 2)
        w, h = rng.uniform(0.05, 0.3, 2)
        objects.append((np.array([x1, y1, x1 + w, y1 + h]), int(rng.integers(n_classes))))
    boxes, scores, labels = [], [], []
    for _ in range(n_models):
        b, s, l = [], [], []
        for box, cls in objects:
            for _ in range(rng.integers(*boxes_per_object)):
                b.append(np.clip(box + rng.normal(0, 0.01, 4), 0, 1))
                s.append(rng.uniform(0.05, 0.99))
                l.append(cls)
        boxes.append(np.array(b).reshape(-1, 4))
        scores.append(np.array(s))
        labels.append(np.array(l))
    return boxes, scores, labels


def sort_by_score(b, s, l):
    o = np.lexsort((b[:, 0], -np.round(s, 6)))
    return b[o], s[o], l[o]


@pytest.mark.parametrize("conf_type", ["avg", "max", "box_and_model_avg", "absent_model_aware_avg"])
@pytest.mark.parametrize("seed", range(25))
def test_matches_reference(conf_type, seed):
    rng = np.random.default_rng(seed)
    boxes, scores, labels = random_predictions(rng)
    weights = list(rng.uniform(0.5, 2.0, len(boxes)))
    ref = ensemble_boxes.weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=0.55,
                                               skip_box_thr=0.01, conf_type=conf_type)
    ours = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=0.55,
                                 skip_box_thr=0.01, conf_type=conf_type)
    assert len(ours[0]) == len(ref[0])
    rb, rs, rl = sort_by_score(np.asarray(ref[0]), np.asarray(ref[1]), np.asarray(ref[2]))
    ob, os_, ol = sort_by_score(*ours)
    np.testing.assert_allclose(os_, rs, atol=1e-5)
    np.testing.assert_allclose(ob, rb, atol=1e-5)
    np.testing.assert_array_equal(ol, rl.astype(int))


@pytest.mark.parametrize("seed", range(25))
def test_avg_score_never_exceeds_one(seed):
    """The old implementation summed duplicate boxes from one model and could return scores > 1."""
    rng = np.random.default_rng(100 + seed)
    boxes, scores, labels = random_predictions(rng, boxes_per_object=(1, 4))
    for conf_type in ("avg", "consensus_boosted"):
        _, s, _ = weighted_boxes_fusion(boxes, scores, labels, weights=[1.0, 1.1, 1.1], conf_type=conf_type)
        assert (s <= 1.0 + 1e-6).all()


def test_class_conf_thresholds_filter_only_their_class():
    boxes = [np.array([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.7, 0.7]])]
    scores = [np.array([0.2, 0.2])]
    labels = [np.array([0, 1])]
    _, s, l = weighted_boxes_fusion(boxes, scores, labels, class_conf_thrs={0: 0.5})
    assert list(l) == [1]


def test_empty_input():
    b, s, l = weighted_boxes_fusion([np.zeros((0, 4))] * 2, [np.zeros(0)] * 2, [np.zeros(0)] * 2)
    assert b.shape == (0, 4) and s.shape == (0,) and l.shape == (0,)

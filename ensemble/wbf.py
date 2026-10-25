"""
Weighted Boxes Fusion (WBF) Implementation for Multi-Model Object Detection Ensemble.

Reference:
    Solovyev, R., Wang, W., & Gabruseva, T. (2021).
    "Weighted boxes fusion: Ensembling boxes from different object detection models."
    Image and Vision Computing, 107, 104117.

This is a standalone, dependency-free implementation optimized for the SWORD weapon
detection pipeline. It supports multi-class detections, configurable model weights,
IoU-based box grouping, and coordinate normalization.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    
    return inter / union if union > 0 else 0.0


def _iou_vectorized(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU of one box against a set of boxes. All [x1, y1, x2, y2]."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    
    return np.where(union > 0, inter / union, 0.0)


def _fuse_rows(rows: np.ndarray, conf_type: str) -> np.ndarray:
    """Fuse cluster rows [weighted_score, weight, model, x1, y1, x2, y2] into one row of the same layout.

    Coordinates are averaged with the weighted scores as weights; the score field holds the mean weighted
    score ("max" conf_type: the maximum), exactly as in the reference implementation.
    """
    ws = rows[:, 0]
    fused = np.zeros(7, dtype=np.float64)
    fused[3:7] = (ws[:, None] * rows[:, 3:7]).sum(axis=0) / ws.sum()
    fused[0] = ws.max() if conf_type == "max" else ws.sum() / len(rows)
    fused[1] = rows[:, 1].sum()
    return fused


def weighted_boxes_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    class_weights: Optional[Dict[int, List[float]]] = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
    conf_type: str = "avg",
    allows_overflow: bool = False,
    consensus_boost: float = 0.15,
    outlier_thr: float = 0.35,
    outlier_penalty: float = 0.65,
    class_conf_thrs: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted Boxes Fusion (WBF) for ensembling predictions from multiple models.

    The confidence types "avg", "max", "box_and_model_avg" and "absent_model_aware_avg" reproduce the
    reference implementation by the WBF authors (``ensemble_boxes.weighted_boxes_fusion``) exactly;
    ``tests/test_wbf.py`` checks parity on random inputs. In particular, a cluster's "avg" score is
    ``mean(score * w) * min(n_models, n_boxes) / sum(w)``, so it can no longer exceed 1 when one model
    contributes several boxes to the same cluster (it could before).

    Project extensions on top of the reference:
        * class_weights: per-class model weights (override ``weights`` for the given class IDs).
        * conf_type="consensus_boosted": "avg" score, boosted when every model contributes and penalised
          when a single model contributes a low score; always clipped to 1.
        * class_conf_thrs: per-class minimum score for a fused box to be returned.

    Args:
        boxes_list: per model, (N_i, 4) arrays of [x1, y1, x2, y2] **normalised to [0, 1]**.
        scores_list: per model, (N_i,) confidence scores.
        labels_list: per model, (N_i,) integer class labels.
        weights: per-model weights (default 1.0 each).
        class_weights: optional {class_id: [w_model_0, w_model_1, ...]}.
        iou_thr: a box joins a cluster when its IoU with the cluster's fused box is > iou_thr.
        skip_box_thr: boxes scoring below this are ignored.
        conf_type: "avg", "max", "box_and_model_avg", "absent_model_aware_avg" or "consensus_boosted".
        allows_overflow: reference semantics — if True, the "avg" score is not capped by min(n_models, n_boxes).
        consensus_boost, outlier_thr, outlier_penalty: parameters of "consensus_boosted".
        class_conf_thrs: optional {class_id: min_score}.

    Returns:
        (fused_boxes (M, 4), fused_scores (M,), fused_labels (M,)), sorted by score, descending.
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models
    if len(weights) != n_models:
        raise ValueError(f"weights length ({len(weights)}) must match number of models ({n_models})")
    if conf_type not in {"avg", "max", "box_and_model_avg", "absent_model_aware_avg", "consensus_boosted"}:
        raise ValueError(f"Unknown conf_type: {conf_type}")
    global_w = np.asarray(weights, dtype=np.float64)
    cls_w = {}
    for c, cw in (class_weights or {}).items():
        if len(cw) != n_models:
            raise ValueError(f"class_weights[{c}] length must match number of models ({n_models})")
        cls_w[int(c)] = np.asarray(cw, dtype=np.float64)

    # Pre-filter (as in the reference): drop low scores, order and clip coordinates, drop zero-area boxes.
    per_label: Dict[int, list] = {}
    for m in range(n_models):
        boxes = np.asarray(boxes_list[m], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores_list[m], dtype=np.float64).reshape(-1)
        labels = np.asarray(labels_list[m]).reshape(-1)
        if len(boxes) == 0:
            continue
        keep = scores >= skip_box_thr
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        x1 = np.clip(np.minimum(boxes[:, 0], boxes[:, 2]), 0.0, 1.0)
        x2 = np.clip(np.maximum(boxes[:, 0], boxes[:, 2]), 0.0, 1.0)
        y1 = np.clip(np.minimum(boxes[:, 1], boxes[:, 3]), 0.0, 1.0)
        y2 = np.clip(np.maximum(boxes[:, 1], boxes[:, 3]), 0.0, 1.0)
        for i in np.nonzero((x2 - x1) * (y2 - y1) > 0)[0]:
            lab = int(labels[i])
            w = cls_w.get(lab, global_w)[m]
            per_label.setdefault(lab, []).append([scores[i] * w, w, m, x1[i], y1[i], x2[i], y2[i]])

    fused_boxes, fused_scores, fused_labels = [], [], []
    for lab, rows in per_label.items():
        arr = np.asarray(rows, dtype=np.float64)
        arr = arr[np.argsort(-arr[:, 0], kind="stable")]
        w_lab = cls_w.get(lab, global_w)
        clusters: List[List[int]] = []
        fused = np.empty((0, 7), dtype=np.float64)
        for j in range(len(arr)):
            if len(fused):
                ious = _iou_vectorized(arr[j, 3:7], fused[:, 3:7])
                k = int(np.argmax(ious))
                if ious[k] > iou_thr:
                    clusters[k].append(j)
                    fused[k] = _fuse_rows(arr[clusters[k]], conf_type)
                    continue
            clusters.append([j])
            fused = np.vstack([fused, arr[j]])

        for k, members in enumerate(clusters):
            rows_k = arr[members]
            n = len(rows_k)
            f = fused[k]
            if conf_type == "box_and_model_avg":
                score = f[0] * n / rows_k[:, 1].sum()
                _, first = np.unique(rows_k[:, 2], return_index=True)
                score = score * rows_k[first, 1].sum() / w_lab.sum()
            elif conf_type == "absent_model_aware_avg":
                absent = np.ones(n_models, dtype=bool)
                absent[np.unique(rows_k[:, 2]).astype(int)] = False
                score = f[0] * n / (rows_k[:, 1].sum() + w_lab[absent].sum())
            elif conf_type == "max":
                score = f[0] / w_lab.max()
            else:  # "avg" and "consensus_boosted"
                score = f[0] * (n if allows_overflow else min(n_models, n)) / w_lab.sum()
                if conf_type == "consensus_boosted" and n_models > 1:
                    n_contrib = len(np.unique(rows_k[:, 2]))
                    if n_contrib == n_models:
                        score = min(1.0, score * (1.0 + consensus_boost))
                    elif n_contrib == 1 and score < outlier_thr:
                        score *= outlier_penalty
            if class_conf_thrs is not None and lab in class_conf_thrs and score < class_conf_thrs[lab]:
                continue
            fused_boxes.append(f[3:7])
            fused_scores.append(score)
            fused_labels.append(lab)

    if not fused_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    order = np.argsort(-np.asarray(fused_scores), kind="stable")
    return (
        np.asarray(fused_boxes, dtype=np.float32)[order],
        np.asarray(fused_scores, dtype=np.float32)[order],
        np.asarray(fused_labels, dtype=np.int32)[order],
    )


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Non-Maximum Suppression (NMS) for single-model post-processing.
    
    Args:
        boxes: (N, 4) array [x1, y1, x2, y2] in pixel or normalized coords.
        scores: (N,) confidence scores.
        labels: (N,) class labels.
        iou_thr: IoU threshold to suppress overlapping boxes.
    
    Returns:
        Tuple of filtered (boxes, scores, labels).
    """
    if len(boxes) == 0:
        return boxes, scores, labels
    
    keep = []
    unique_labels = np.unique(labels)
    
    for cls in unique_labels:
        cls_mask = labels == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        order = np.argsort(-cls_scores)
        cls_indices = cls_indices[order]
        cls_boxes = cls_boxes[order]
        
        suppressed = np.zeros(len(cls_boxes), dtype=bool)
        
        for i in range(len(cls_boxes)):
            if suppressed[i]:
                continue
            keep.append(cls_indices[i])
            
            if i + 1 < len(cls_boxes):
                remaining = cls_boxes[i + 1:]
                remaining_mask = ~suppressed[i + 1:]
                if remaining_mask.any():
                    ious = _iou_vectorized(cls_boxes[i], remaining)
                    for j in range(len(remaining)):
                        if not suppressed[i + 1 + j] and ious[j] >= iou_thr:
                            suppressed[i + 1 + j] = True
    
    if len(keep) == 0:
        return (
            np.zeros((0, 4), dtype=boxes.dtype),
            np.zeros((0,), dtype=scores.dtype),
            np.zeros((0,), dtype=labels.dtype),
        )
    
    keep = np.array(keep)
    return boxes[keep], scores[keep], labels[keep]

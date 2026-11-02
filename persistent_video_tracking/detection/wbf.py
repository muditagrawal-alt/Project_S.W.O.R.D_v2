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
    
    Supports global model weights, class-adaptive model weights, consensus-boosted
    confidence scoring, and per-class confidence threshold filtering.
    
    Args:
        boxes_list: List of arrays, each shape (N_i, 4) with [x1, y1, x2, y2]
                    coordinates **normalized to [0, 1]**.
        scores_list: List of arrays, each shape (N_i,) with confidence scores.
        labels_list: List of arrays, each shape (N_i,) with integer class labels.
        weights: Global per-model weight factors. Default: equal weights (1.0 each).
        class_weights: Optional dict mapping class_id -> list of weights per model.
                       Overrides global weights for specified classes.
        iou_thr: IoU threshold for grouping boxes as belonging to the same object.
        skip_box_thr: Minimum score threshold; discard boxes below this before fusion.
        conf_type: How to compute fused confidence:
            - "avg": Average confidence over all models.
            - "consensus_boosted": Boosts confidence when all models agree, penalizes isolated low-conf detections.
            - "max": Maximum confidence of any contributing box.
            - "box_and_model_avg": Weighted average factoring in model weights.
            - "absent_model_aware_avg": Average that accounts for models that didn't detect the object.
        allows_overflow: If True, don't clip box coordinates to [0, 1].
        consensus_boost: Multiplicative boost factor for full consensus (e.g. 0.15 = +15%).
        outlier_thr: Threshold below which single-model detections are penalized as outliers.
        outlier_penalty: Multiplicative penalty for single-model outlier detections (e.g. 0.65).
        class_conf_thrs: Optional dict mapping class_id -> min confidence threshold for output.
    
    Returns:
        Tuple of (fused_boxes, fused_scores, fused_labels):
            - fused_boxes: (M, 4) array of merged bounding boxes.
            - fused_scores: (M,) array of fused confidence scores.
            - fused_labels: (M,) array of class labels.
    """
    n_models = len(boxes_list)
    
    if weights is None:
        weights = [1.0] * n_models
    
    assert len(weights) == n_models, (
        f"weights length ({len(weights)}) must match number of models ({n_models})"
    )
    
    # Normalize global weights so they sum to N_models (preserves averaging semantics)
    weight_sum = sum(weights)
    norm_global_weights = [w * n_models / weight_sum for w in weights] if weight_sum > 0 else [1.0] * n_models
    
    # Normalize per-class weights if provided
    norm_class_weights = {}
    if class_weights is not None:
        for c, c_w in class_weights.items():
            assert len(c_w) == n_models, f"class_weights[{c}] length must match {n_models}"
            c_sum = sum(c_w)
            norm_class_weights[c] = [w * n_models / c_sum for w in c_w] if c_sum > 0 else norm_global_weights

    # Collect all boxes with metadata: [x1, y1, x2, y2, score, model_idx, original_idx]
    all_boxes = []
    all_scores = []
    all_labels = []
    all_model_idx = []
    
    for model_i in range(n_models):
        boxes = boxes_list[model_i]
        scores = scores_list[model_i]
        labels = labels_list[model_i]
        
        if len(boxes) == 0:
            continue
        
        # Filter by skip_box_thr
        mask = scores >= skip_box_thr
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        if len(boxes) == 0:
            continue
            
        # Apply class-specific or global model weights
        if norm_class_weights:
            weighted_scores = np.empty_like(scores)
            for i in range(len(scores)):
                cls_id = int(labels[i])
                cls_w = norm_class_weights.get(cls_id, norm_global_weights)
                weighted_scores[i] = scores[i] * cls_w[model_i]
        else:
            weighted_scores = scores * norm_global_weights[model_i]
        
        all_boxes.append(boxes)
        all_scores.append(weighted_scores)
        all_labels.append(labels)
        all_model_idx.append(np.full(len(boxes), model_i, dtype=np.int32))
    
    if len(all_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_model_idx = np.concatenate(all_model_idx, axis=0)
    
    # Process per class
    unique_labels = np.unique(all_labels)
    fused_boxes = []
    fused_scores = []
    fused_labels_out = []
    
    for cls in unique_labels:
        cls_mask = all_labels == cls
        cls_boxes = all_boxes[cls_mask].copy()
        cls_scores = all_scores[cls_mask].copy()
        cls_model_idx = all_model_idx[cls_mask].copy()
        
        if len(cls_boxes) == 0:
            continue
        
        # Sort by score descending
        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_model_idx = cls_model_idx[order]
        
        # Cluster boxes using greedy IoU-based grouping
        # Each cluster will be fused into a single output box
        clusters = []  # List of lists of (box, score, model_idx)
        cluster_boxes = []  # Representative box for each cluster (running weighted avg)
        
        for i in range(len(cls_boxes)):
            box = cls_boxes[i]
            score = cls_scores[i]
            model_i = cls_model_idx[i]
            
            matched_cluster = -1
            best_iou = iou_thr
            
            if len(cluster_boxes) > 0:
                ious = _iou_vectorized(box, np.array(cluster_boxes))
                max_iou_idx = np.argmax(ious)
                if ious[max_iou_idx] >= iou_thr:
                    matched_cluster = max_iou_idx
                    best_iou = ious[max_iou_idx]
            
            if matched_cluster >= 0:
                clusters[matched_cluster].append((box.copy(), score, model_i))
                # Update cluster representative as weighted average
                members = clusters[matched_cluster]
                weighted_coords = np.zeros(4)
                total_weight = 0.0
                for mb, ms, _ in members:
                    weighted_coords += mb * ms
                    total_weight += ms
                if total_weight > 0:
                    cluster_boxes[matched_cluster] = weighted_coords / total_weight
            else:
                clusters.append([(box.copy(), score, model_i)])
                cluster_boxes.append(box.copy())
        
        # Compute fused output for each cluster
        for cluster in clusters:
            n_boxes = len(cluster)
            
            # Weighted average of coordinates
            weighted_coords = np.zeros(4)
            total_conf = 0.0
            max_conf = 0.0
            model_contributions = {}
            
            for mb, ms, mi in cluster:
                weighted_coords += mb * ms
                total_conf += ms
                max_conf = max(max_conf, ms)
                model_contributions[mi] = model_contributions.get(mi, 0.0) + ms
            
            if total_conf > 0:
                merged_box = weighted_coords / total_conf
            else:
                merged_box = cluster[0][0]
            
            # Fused confidence score
            if conf_type == "avg":
                # Average over all models (penalizes if not all models agree)
                merged_score = total_conf / n_models
            elif conf_type == "consensus_boosted":
                n_contributing = len(model_contributions)
                base_score = total_conf / n_models
                if n_contributing >= n_models and n_models > 1:
                    # Consensus boost when all models detect
                    merged_score = min(1.0, base_score * (1.0 + consensus_boost))
                elif n_contributing == 1 and n_models > 1:
                    # Single model detection
                    if base_score < outlier_thr:
                        merged_score = base_score * outlier_penalty
                    else:
                        merged_score = base_score
                else:
                    merged_score = base_score
            elif conf_type == "max":
                merged_score = max_conf
            elif conf_type == "box_and_model_avg":
                # Average confidence within each model, then average across models
                per_model_avg = []
                for mi in range(n_models):
                    if mi in model_contributions:
                        model_boxes = [ms for _, ms, m in cluster if m == mi]
                        per_model_avg.append(np.mean(model_boxes))
                if len(per_model_avg) > 0:
                    merged_score = np.mean(per_model_avg)
                else:
                    merged_score = total_conf / n_models
            elif conf_type == "absent_model_aware_avg":
                # Like avg, but only divides by number of models that contributed
                n_contributing = len(model_contributions)
                merged_score = total_conf / max(n_contributing, 1)
            else:
                raise ValueError(f"Unknown conf_type: {conf_type}")
            
            # Apply per-class confidence threshold if specified
            if class_conf_thrs is not None and cls in class_conf_thrs:
                if merged_score < class_conf_thrs[cls]:
                    continue

            if not allows_overflow:
                merged_box = np.clip(merged_box, 0.0, 1.0)
            
            fused_boxes.append(merged_box)
            fused_scores.append(merged_score)
            fused_labels_out.append(cls)
    
    if len(fused_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    
    fused_boxes = np.array(fused_boxes, dtype=np.float32)
    fused_scores = np.array(fused_scores, dtype=np.float32)
    fused_labels_out = np.array(fused_labels_out, dtype=np.int32)
    
    # Final sort by score
    order = np.argsort(-fused_scores)
    return fused_boxes[order], fused_scores[order], fused_labels_out[order]


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

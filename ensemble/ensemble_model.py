"""
SWORD v2 Ensemble Detector: 3-Model WBF Fusion (YOLO26s + YOLO26m + RT-DETR-L).

This module provides a unified interface for multi-model ensemble inference
on the SWORD weapon detection dataset. It loads up to three models, runs
inference on each independently, then fuses predictions using Weighted Boxes
Fusion (WBF) to produce higher-quality detections.

The ensemble combines complementary strengths:
    - YOLO26s: Higher recall, fast inference
    - YOLO26m: Higher precision, stronger features
    - RT-DETR-L: Best standalone mAP, transformer-based attention

Usage:
    from ensemble.ensemble_model import SWORDEnsembleDetector

    # 3-model ensemble (recommended)
    detector = SWORDEnsembleDetector(
        weights_s="weights/yolo26s_best.pt",
        weights_m="weights/yolo26m_best.pt",
        weights_rtdetr="weights/rtdetr_l_best.pt",
    )

    # 2-model ensemble (YOLO only)
    detector = SWORDEnsembleDetector(
        weights_s="weights/yolo26s_best.pt",
        weights_m="weights/yolo26m_best.pt",
    )

    results = detector.predict("path/to/image.jpg")
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

from ensemble.wbf import weighted_boxes_fusion, nms


# SWORD v2 class names (10 weapon classes)
CLASS_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]


class SWORDEnsembleDetector:
    """
    Multi-model ensemble detector using Weighted Boxes Fusion.

    Supports 2-model (YOLO26s + YOLO26m) or 3-model (+ RT-DETR-L) configurations.

    Attributes:
        models: List of loaded model instances.
        model_names: Human-readable names for each model.
        model_weights: Per-model weights for WBF fusion.
        iou_thr: IoU threshold for WBF box grouping.
        conf_thr: Minimum confidence threshold for output detections.
        fusion_mode: "wbf" (default), "nms", or "soft_nms".
    """

    def __init__(
        self,
        weights_s: str,
        weights_m: str,
        weights_rtdetr: Optional[str] = None,
        model_weights: Optional[Tuple[float, ...]] = None,
        iou_thr: float = 0.55,
        conf_thr: float = 0.25,
        fusion_mode: str = "wbf",
        device: Optional[str] = None,
        params_file: Optional[str] = None,
        conf_type: str = "avg",
        consensus_boost: float = 0.15,
        outlier_thr: float = 0.35,
        outlier_penalty: float = 0.65,
        class_weights: Optional[Dict[int, List[float]]] = None,
        class_conf_thrs: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize the ensemble detector.

        Args:
            weights_s: Path to YOLO26s fine-tuned weights.
            weights_m: Path to YOLO26m fine-tuned weights.
            weights_rtdetr: Path to RT-DETR-L fine-tuned weights (optional; omit for 2-model ensemble).
            model_weights: Per-model WBF weights. Default: (1.0, 1.1) for 2-model,
                           (1.0, 1.1, 1.1) for 3-model.
            iou_thr: IoU threshold for WBF grouping.
            conf_thr: Minimum confidence for final output detections.
            fusion_mode: "wbf" or "nms".
            device: "cuda", "mps", "cpu", or None (auto-detect).
            params_file: Optional path to JSON file containing optimized non-training parameters.
            conf_type: "avg" or "consensus_boosted".
            consensus_boost: Multiplicative boost factor for full consensus.
            outlier_thr: Score cutoff below which single-model detections are penalized.
            outlier_penalty: Multiplicative penalty for single-model outliers.
            class_weights: Optional per-class model weights dict.
            class_conf_thrs: Optional per-class output confidence threshold dict.
        """
        import json
        from ultralytics import YOLO, RTDETR

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.fusion_mode = fusion_mode
        self.conf_type = conf_type
        self.consensus_boost = consensus_boost
        self.outlier_thr = outlier_thr
        self.outlier_penalty = outlier_penalty
        self.class_weights = class_weights
        self.class_conf_thrs = class_conf_thrs
        self.models = []
        self.model_names = []

        # Load YOLO models
        print(f"Loading YOLO26s from {weights_s}...")
        self.models.append(YOLO(weights_s))
        self.model_names.append("YOLO26s")

        print(f"Loading YOLO26m from {weights_m}...")
        self.models.append(YOLO(weights_m))
        self.model_names.append("YOLO26m")

        # Optionally load RT-DETR-L
        if weights_rtdetr is not None:
            print(f"Loading RT-DETR-L from {weights_rtdetr}...")
            self.models.append(RTDETR(weights_rtdetr))
            self.model_names.append("RT-DETR-L")

        # Set model weights
        n = len(self.models)
        if model_weights is not None:
            assert len(model_weights) == n, (
                f"model_weights length ({len(model_weights)}) must match number of models ({n})"
            )
            self.model_weights = list(model_weights)
        else:
            # Default weights: slightly favor larger models
            self.model_weights = [1.0, 1.1, 1.1][:n]

        # Load optimized parameters JSON if supplied or auto-discover
        if params_file is not None and Path(params_file).exists():
            with open(params_file) as fp:
                p_data = json.load(fp)
            if "class_weights" in p_data and n == 3:
                self.class_weights = {int(k): v for k, v in p_data["class_weights"].items()}
            if "class_conf_thrs" in p_data:
                self.class_conf_thrs = {int(k): v for k, v in p_data["class_conf_thrs"].items()}
            if "iou_thr" in p_data:
                self.iou_thr = p_data["iou_thr"]
            if "conf_type" in p_data:
                self.conf_type = p_data["conf_type"]
            if "consensus_boost" in p_data:
                self.consensus_boost = p_data["consensus_boost"]
            if "outlier_thr" in p_data:
                self.outlier_thr = p_data["outlier_thr"]
            if "outlier_penalty" in p_data:
                self.outlier_penalty = p_data["outlier_penalty"]
            print(f"  Loaded optimized parameters from: {params_file}")

        print(f"\nEnsemble ready ({n} models): {', '.join(self.model_names)}")
        print(f"  mode={fusion_mode}, conf_type={self.conf_type}, weights={self.model_weights}, "
              f"iou_thr={self.iou_thr}, conf_thr={self.conf_thr}, device={self.device}")

    @property
    def n_models(self) -> int:
        """Number of models in the ensemble."""
        return len(self.models)

    def _run_single_model(
        self,
        model,
        source: Union[str, np.ndarray],
        img_size: int = 640,
        conf: float = 0.001,
        iou: float = 0.7,
    ) -> List[Dict]:
        """
        Run inference with a single model and extract raw detections.

        Returns:
            List of dicts per image, each containing:
                - "boxes": (N, 4) array [x1, y1, x2, y2] in pixel coords
                - "scores": (N,) confidence scores
                - "labels": (N,) integer class labels
                - "img_shape": (height, width) of the original image
        """
        results = model.predict(
            source=source,
            imgsz=img_size,
            conf=conf,
            iou=iou,
            verbose=False,
            device=self.device,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                det = {
                    "boxes": boxes.xyxy.cpu().numpy().astype(np.float32),
                    "scores": boxes.conf.cpu().numpy().astype(np.float32),
                    "labels": boxes.cls.cpu().numpy().astype(np.int32),
                    "img_shape": r.orig_shape,  # (H, W)
                }
            else:
                det = {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                    "labels": np.zeros((0,), dtype=np.int32),
                    "img_shape": r.orig_shape,
                }
            detections.append(det)

        return detections

    def predict(
        self,
        source: Union[str, np.ndarray, List[str]],
        img_size: int = 640,
        raw_conf: float = 0.001,
        raw_iou: float = 0.7,
    ) -> List[Dict]:
        """
        Run ensemble inference on one or more images.

        Args:
            source: Path to image, numpy array, or list of paths.
            img_size: Input image size for inference.
            raw_conf: Low confidence threshold for individual models (filtered later via WBF).
            raw_iou: NMS IoU threshold within each model's internal post-processing.

        Returns:
            List of dicts per image with fused detections:
                - "boxes": (M, 4) pixel-coordinate bounding boxes
                - "scores": (M,) fused confidence scores
                - "labels": (M,) class labels
                - "img_shape": (H, W)
        """
        # Run all models
        all_model_dets = []
        for model in self.models:
            dets = self._run_single_model(model, source, img_size, raw_conf, raw_iou)
            all_model_dets.append(dets)

        n_images = len(all_model_dets[0])
        for dets in all_model_dets[1:]:
            assert len(dets) == n_images, "Model outputs must have same number of images"

        fused_results = []
        for img_idx in range(n_images):
            h, w = all_model_dets[0][img_idx]["img_shape"]

            # Collect and normalize boxes from all models
            boxes_list = []
            scores_list = []
            labels_list = []

            for model_dets in all_model_dets:
                det = model_dets[img_idx]
                boxes_norm = det["boxes"].copy()
                if len(boxes_norm) > 0:
                    boxes_norm[:, [0, 2]] /= w
                    boxes_norm[:, [1, 3]] /= h
                boxes_list.append(boxes_norm)
                scores_list.append(det["scores"])
                labels_list.append(det["labels"])

            if self.fusion_mode == "wbf":
                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                    boxes_list=boxes_list,
                    scores_list=scores_list,
                    labels_list=labels_list,
                    weights=self.model_weights,
                    class_weights=self.class_weights,
                    iou_thr=self.iou_thr,
                    skip_box_thr=0.001,
                    conf_type=self.conf_type,
                    consensus_boost=self.consensus_boost,
                    outlier_thr=self.outlier_thr,
                    outlier_penalty=self.outlier_penalty,
                    class_conf_thrs=self.class_conf_thrs,
                )
            elif self.fusion_mode == "nms":
                all_boxes = np.concatenate(boxes_list, axis=0)
                all_scores = np.concatenate(scores_list, axis=0)
                all_labels = np.concatenate(labels_list, axis=0)
                fused_boxes, fused_scores, fused_labels = nms(
                    all_boxes, all_scores, all_labels, iou_thr=self.iou_thr
                )
            else:
                raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

            # Apply global confidence threshold fallback if class_conf_thrs was not set
            if self.class_conf_thrs is None:
                mask = fused_scores >= self.conf_thr
                fused_boxes = fused_boxes[mask]
                fused_scores = fused_scores[mask]
                fused_labels = fused_labels[mask]

            # Denormalize back to pixel coordinates
            if len(fused_boxes) > 0:
                fused_boxes[:, [0, 2]] *= w
                fused_boxes[:, [1, 3]] *= h

            fused_results.append({
                "boxes": fused_boxes,
                "scores": fused_scores,
                "labels": fused_labels,
                "img_shape": (h, w),
            })

        return fused_results

    def predict_and_compare(
        self,
        source: Union[str, np.ndarray],
        img_size: int = 640,
    ) -> Dict:
        """
        Run inference and return individual + ensemble results for comparison.

        Returns:
            Dict with keys for each model name and "ensemble", each containing
            boxes/scores/labels/img_shape.
        """
        individual_results = {}
        for model, name in zip(self.models, self.model_names):
            dets = self._run_single_model(model, source, img_size)
            individual_results[name] = dets[0]

        # Get fused result
        fused = self.predict(source, img_size)

        result = dict(individual_results)
        result["ensemble"] = fused[0]
        return result

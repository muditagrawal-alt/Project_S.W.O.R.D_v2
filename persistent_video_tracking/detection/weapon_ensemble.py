"""
Weapon Ensemble Detector for S.W.O.R.D Phase 2.
Fuses predictions from YOLO26s and YOLO26m using Weighted Boxes Fusion (WBF)
before passing detections to ByteTrack.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

# Import WBF implementation with robust fallback
try:
    from .wbf import weighted_boxes_fusion, nms
except (ImportError, ValueError):
    try:
        from persistent_video_tracking.detection.wbf import weighted_boxes_fusion, nms
    except (ImportError, ValueError):
        ROOT_DIR = Path(__file__).resolve().parent.parent.parent
        if str(ROOT_DIR) not in sys.path:
            sys.path.insert(0, str(ROOT_DIR))
        from ensemble.wbf import weighted_boxes_fusion, nms

# Canonical 10 SWORD v2 weapon class names
SWORD_WEAPON_CLASSES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun"
]


def resolve_model_path(path_str: str) -> str:
    """Resolve model path relative to CWD, repository root, or package directory."""
    path = Path(path_str)
    if path.is_file():
        return str(path.resolve())

    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / path_str
    if candidate.is_file():
        return str(candidate.resolve())

    pkg_root = Path(__file__).resolve().parent.parent
    candidate_pkg = pkg_root / path_str
    if candidate_pkg.is_file():
        return str(candidate_pkg.resolve())

    return str(path)


class WeaponEnsembleDetector:
    """
    Two-model weapon detector ensemble (YOLO26s + YOLO26m) using Weighted Boxes Fusion.
    Preserves model provenance, executes both models independently, and fuses boxes.
    """

    def __init__(
        self,
        weights_s: str = "weights/yolo26s_best.pt",
        weights_m: str = "weights/yolo26m_best.pt",
        model_weights: Tuple[float, float] = (1.0, 1.1),
        conf_thr: float = 0.25,
        iou_thr: float = 0.55,
        img_size: int = 640,
        fusion_mode: str = "wbf",
        conf_type: str = "avg",
        device: Optional[str] = "cpu",
    ):
        self.weights_s_path = resolve_model_path(weights_s)
        self.weights_m_path = resolve_model_path(weights_m)
        self.model_weights = list(model_weights)
        self.conf_thr = float(conf_thr)
        self.iou_thr = float(iou_thr)
        self.img_size = int(img_size)
        self.fusion_mode = str(fusion_mode).lower()
        self.conf_type = str(conf_type).lower()
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.weights_s_path):
            raise FileNotFoundError(f"YOLO26s weights not found: {self.weights_s_path}")
        if not os.path.exists(self.weights_m_path):
            raise FileNotFoundError(f"YOLO26m weights not found: {self.weights_m_path}")


        print(f"[WeaponEnsemble] Loading YOLO26s from '{self.weights_s_path}' (device={self.device})...", flush=True)
        self.model_s = YOLO(self.weights_s_path)
        print(f"[WeaponEnsemble] Loading YOLO26m from '{self.weights_m_path}' (device={self.device})...", flush=True)
        self.model_m = YOLO(self.weights_m_path)

        # Build class names lookup
        if hasattr(self.model_m, "names") and self.model_m.names:
            self.class_names = self.model_m.names
        elif hasattr(self.model_s, "names") and self.model_s.names:
            self.class_names = self.model_s.names
        else:
            self.class_names = {i: name for i, name in enumerate(SWORD_WEAPON_CLASSES)}

        print(
            f"[WeaponEnsemble] Ready: 2 models (YOLO26s, YOLO26m), weights={self.model_weights}, "
            f"conf_thr={self.conf_thr}, iou_thr={self.iou_thr}, mode={self.fusion_mode}",
            flush=True
        )

    def get_provenance(self) -> Dict:
        """Return full provenance metadata for research reproducibility."""
        return {
            "detector_type": "YOLO_Ensemble_WBF",
            "models": [
                {"name": "YOLO26s", "path": self.weights_s_path, "ensemble_weight": self.model_weights[0]},
                {"name": "YOLO26m", "path": self.weights_m_path, "ensemble_weight": self.model_weights[1]},
            ],
            "fusion_mode": self.fusion_mode,
            "conf_thr": self.conf_thr,
            "iou_thr": self.iou_thr,
            "img_size": self.img_size,
            "device": self.device,
            "class_names": self.class_names,
        }

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference with both YOLO26s and YOLO26m and fuse detections.

        Args:
            frame: BGR numpy image frame from OpenCV.

        Returns:
            sv.Detections object with fused xyxy, confidence, and class_id.
        """
        h, w = frame.shape[:2]

        # 1. Run YOLO26s inference
        res_s = self.model_s.predict(
            source=frame,
            imgsz=self.img_size,
            conf=0.001,  # low raw threshold to retain candidates for WBF
            iou=0.7,
            verbose=False,
            device=self.device
        )[0]

        # 2. Run YOLO26m inference
        res_m = self.model_m.predict(
            source=frame,
            imgsz=self.img_size,
            conf=0.001,
            iou=0.7,
            verbose=False,
            device=self.device
        )[0]

        # Extract normalized boxes, scores, and class labels
        boxes_list = []
        scores_list = []
        labels_list = []

        for res in (res_s, res_m):
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                conf = boxes.conf.cpu().numpy().astype(np.float32)
                cls_id = boxes.cls.cpu().numpy().astype(np.int32)

                # Normalize to [0, 1] for WBF
                norm_xyxy = xyxy.copy()
                norm_xyxy[:, [0, 2]] /= max(w, 1)
                norm_xyxy[:, [1, 3]] /= max(h, 1)

                boxes_list.append(norm_xyxy)
                scores_list.append(conf)
                labels_list.append(cls_id)
            else:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                scores_list.append(np.zeros((0,), dtype=np.float32))
                labels_list.append(np.zeros((0,), dtype=np.int32))

        # Check if there are any detections across both models
        total_raw_boxes = sum(len(b) for b in boxes_list)
        if total_raw_boxes == 0:
            return sv.Detections.empty()

        # 3. Fuse with WBF or NMS
        if self.fusion_mode == "wbf":
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list=boxes_list,
                scores_list=scores_list,
                labels_list=labels_list,
                weights=self.model_weights,
                iou_thr=self.iou_thr,
                skip_box_thr=0.001,
                conf_type=self.conf_type,
            )
        elif self.fusion_mode == "nms":
            all_boxes = np.concatenate(boxes_list, axis=0)
            all_scores = np.concatenate(scores_list, axis=0)
            all_labels = np.concatenate(labels_list, axis=0)
            fused_boxes, fused_scores, fused_labels = nms(
                all_boxes, all_scores, all_labels, iou_thr=self.iou_thr
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")

        # 4. Filter by confidence threshold
        mask = fused_scores >= self.conf_thr
        fused_boxes = fused_boxes[mask]
        fused_scores = fused_scores[mask]
        fused_labels = fused_labels[mask]

        if len(fused_boxes) == 0:
            return sv.Detections.empty()

        # 5. Denormalize coordinates back to pixel space
        fused_boxes[:, [0, 2]] *= w
        fused_boxes[:, [1, 3]] *= h

        # Return supervision Detections object
        return sv.Detections(
            xyxy=fused_boxes.astype(np.float32),
            confidence=fused_scores.astype(np.float32),
            class_id=fused_labels.astype(int),
        )

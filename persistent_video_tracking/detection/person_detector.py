"""
Person Detector for S.W.O.R.D Phase 2.
Uses the verified YOLOv8n COCO model strictly for person detection (Class 0).
Kept logically and functionally separate from the 10-class weapon taxonomy.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv


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


class PersonDetector:
    """
    COCO person detector using YOLOv8n.
    Dedicated exclusively to human localization without modifying the weapon taxonomy.
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        conf_thr: float = 0.35,
        img_size: int = 640,
        device: Optional[str] = "cpu",
        coco_person_class_id: int = 0,
    ):
        self.weights_path = resolve_model_path(weights_path)
        self.conf_thr = float(conf_thr)
        self.img_size = int(img_size)
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.person_class_id = int(coco_person_class_id)

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Person detector weights not found: {self.weights_path}")


        print(f"[PersonDetector] Loading YOLOv8n from '{self.weights_path}' (device={self.device})...", flush=True)
        self.model = YOLO(self.weights_path)
        print(f"[PersonDetector] Ready: class_id={self.person_class_id} ('person'), conf_thr={self.conf_thr}", flush=True)

    def get_provenance(self) -> Dict:
        """Return provenance metadata for research reporting."""
        return {
            "detector_type": "YOLOv8n_COCO_Person",
            "model_path": self.weights_path,
            "target_class": "person",
            "coco_class_id": self.person_class_id,
            "conf_thr": self.conf_thr,
            "img_size": self.img_size,
            "device": self.device,
        }

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference and return person detections only.

        Args:
            frame: BGR numpy image frame from OpenCV.

        Returns:
            sv.Detections object containing only person detections.
        """
        # Run inference strictly filtering for person class ID (0 in COCO)
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_thr,
            classes=[self.person_class_id],
            verbose=False,
            device=self.device
        )[0]

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return sv.Detections.empty()

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = boxes.conf.cpu().numpy().astype(np.float32)
        # Person stream keeps a dedicated class_id = 0 representing 'person'
        cls_id = np.zeros(len(xyxy), dtype=int)

        return sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls_id,
        )

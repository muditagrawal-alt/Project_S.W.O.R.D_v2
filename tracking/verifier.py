"""
Second-stage weapon verifier: a small CNN that re-checks every weapon detection on a square crop around it.

The Phase 1 detector confuses hand-held objects (phones, bottles, bags) and scene structure with weapons. The
verifier (ImageNet ResNet-18 fine-tuned on CCTV training crops: true weapons vs the detector's own false detections,
research/tools/phase1/train_weapon_verifier.py) returns p(weapon) per box; the detection score is then fused with
it (fuse_scores). The same crop function is used for training, offline re-scoring and the live system.

    v = WeaponVerifier("weights/weapon_verifier_v1.pt", device="mps")
    p = v.predict(frame_bgr, boxes_xyxy)          # p(weapon) per box
    s = fuse_scores(scores, p, mode="geo")        # sqrt(score * p)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VERIFIER = PROJECT_ROOT / "weights" / "weapon_verifier_v1.pt"
CROP_SCALE, CROP_SIZE, MIN_SIDE = 2.0, 112, 32
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def crop_square(img: np.ndarray, box, scale: float = CROP_SCALE, size: int = CROP_SIZE, jitter=None) -> np.ndarray:
    """Square crop of side scale * max(w, h) around the box centre (zero-padded at the frame edge), resized to size.
    jitter: optional (dx, dy, ds) as fractions of the side, for training augmentation."""
    x1, y1, x2, y2 = map(float, box)
    side = max(MIN_SIDE, scale * max(x2 - x1, y2 - y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if jitter is not None:
        dx, dy, ds = jitter
        cx, cy, side = cx + dx * side, cy + dy * side, side * (1 + ds)
    h, w = img.shape[:2]
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    out = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(0, a), max(0, b), min(w, a + s), min(h, b + s)
    if xb > xa and yb > ya:
        out[ya - b:yb - b, xa - a:xb - a] = img[ya:yb, xa:xb]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)


def to_tensor_batch(crops: np.ndarray):
    import torch

    x = (crops[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD          # BGR -> RGB, ImageNet normalisation
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def build_model():
    import torch.nn as nn
    from torchvision.models import resnet18

    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


class WeaponVerifier:
    def __init__(self, path: str | Path = DEFAULT_VERIFIER, device: str = "cpu"):
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        self.meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        self.scale, self.size = float(self.meta.get("crop_scale", CROP_SCALE)), int(self.meta.get("crop_size", CROP_SIZE))
        self.model = build_model()
        self.model.load_state_dict(ckpt["state_dict"])
        self.device = device
        self.model.to(device).eval()

    def predict(self, img: np.ndarray, boxes) -> np.ndarray:
        import torch

        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros(0, np.float32)
        crops = np.stack([crop_square(img, b, self.scale, self.size) for b in boxes])
        with torch.no_grad():
            p = torch.sigmoid(self.model(to_tensor_batch(crops).to(self.device))).cpu().numpy().reshape(-1)
        return p.astype(np.float32)


def fuse_scores(scores, verify, mode: str = "geo") -> np.ndarray:
    """Detection score fused with the verifier's p(weapon): "geo" sqrt(s * v), "product" s * v, "none" s."""
    s, v = np.asarray(scores, float), np.asarray(verify, float)
    if mode == "none":
        return s
    if mode == "product":
        return s * v
    if mode == "geo":
        return np.sqrt(s * v)
    raise ValueError(f"unknown fusion mode {mode!r}")

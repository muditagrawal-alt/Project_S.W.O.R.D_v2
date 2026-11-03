"""
Person-zoom weapon detection: run the weapon detector on enlarged crops around tracked people and their hands, in
addition to the full frame.

CCTV weapons are tiny (~37x55 px at 1080p, ~12x18 px after the full frame is resized to 640). Phase 2 already
knows where people are and Phase 3 where their wrists are, so a crop around a person (or a hand) resized to the
detector's input size makes a hand-held weapon 2-5x larger. Crop detections are mapped back to frame coordinates and
merged with the full-frame detections (class-wise NMS, highest score kept).

    windows = zoom_windows(persons, frame.shape, ZoomConfig(modes=("person", "hands")))
    boxes, scores, cls = detect_in_windows(model, frame, windows, imgsz=640, conf=0.1, device="mps")
    boxes, scores, cls = merge_detections([(fb, fs, fc), (boxes, scores, cls)])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

L_WRIST, R_WRIST = 9, 10


@dataclass
class ZoomConfig:
    modes: Tuple[str, ...] = ("person",)   # any of "person", "hands"
    person_margin: float = 0.25           # person crop: square of side max(w, h) * (1 + 2 * margin)
    hand_frac: float = 0.6                # hand crop: square of side hand_frac * person height, centred on the wrist
    kpt_thr: float = 0.3                  # wrist keypoint confidence needed for a hand crop
    min_side: int = 64                    # never crop smaller than this (pixels)
    drop_border: bool = True              # drop crop detections cut by a crop edge that is not a frame edge
    max_windows: int = 24                 # cap per frame (crowded scenes)
    min_window: int = 0                   # skip windows whose natural side (before min_side) is smaller: stronger
                                          # magnification than the detector saw in training (smallest crop: 96 px)
    extra: dict = field(default_factory=dict)


def _square(cx: float, cy: float, side: float, W: int, H: int) -> Tuple[int, int, int, int]:
    side = min(side, max(W, H))
    x1, y1 = cx - side / 2, cy - side / 2
    x1 = min(max(0.0, x1), max(0.0, W - side))
    y1 = min(max(0.0, y1), max(0.0, H - side))
    return int(x1), int(y1), int(min(W, x1 + side)), int(min(H, y1 + side))


def zoom_windows(persons: Sequence[dict], frame_shape, cfg: ZoomConfig, with_natural_side: bool = False) -> list:
    """Crop windows (x1, y1, x2, y2) in frame pixels for the tracked persons of one frame. with_natural_side:
    return (window, side before the min_side clamp) pairs instead."""
    H, W = frame_shape[:2]
    out = []
    for p in persons:
        x1, y1, x2, y2 = p["bbox"]
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if "person" in cfg.modes:
            nat = max(w, h) * (1 + 2 * cfg.person_margin)
            out.append((_square((x1 + x2) / 2, (y1 + y2) / 2, max(cfg.min_side, nat), W, H), nat))
        if "hands" in cfg.modes and p.get("keypoints") and p.get("scores"):
            for k in (L_WRIST, R_WRIST):
                if p["scores"][k] >= cfg.kpt_thr:
                    kx, ky = p["keypoints"][k]
                    nat = cfg.hand_frac * h
                    out.append((_square(kx, ky, max(cfg.min_side, nat), W, H), nat))
    out = [(w, n) for w, n in out if w[2] - w[0] >= 8 and w[3] - w[1] >= 8 and n >= cfg.min_window][: cfg.max_windows]
    return out if with_natural_side else [w for w, _ in out]


def from_allowed_window(box, persons: Sequence[dict], frame_shape, cfg: ZoomConfig) -> bool:
    """Offline re-scoring of recorded zoom detections under cfg.min_window: True if the detection's centre lies in a
    window that cfg still allows (windows are recomputed from the recorded persons, so this matches the live rule
    except where an allowed and a skipped window overlap)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(w[0] <= cx <= w[2] and w[1] <= cy <= w[3] for w in zoom_windows(persons, frame_shape, cfg))


def detect_in_windows(model, frame: np.ndarray, windows, imgsz: int = 640, conf: float = 0.1, device=None,
                      drop_border: bool = True):
    """Run `model` (Ultralytics) on each crop; return (boxes[N,4], scores[N], cls[N]) in frame coordinates."""
    if not windows:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    H, W = frame.shape[:2]
    crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in windows]
    results = model.predict(crops, imgsz=imgsz, conf=conf, device=device, verbose=False)
    boxes, scores, cls = [], [], []
    for (x1, y1, x2, y2), r in zip(windows, results):
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        if drop_border and len(b):
            cw, ch, tol = x2 - x1, y2 - y1, 2.0
            cut = (((b[:, 0] <= tol) & (x1 > 0)) | ((b[:, 1] <= tol) & (y1 > 0))
                   | ((b[:, 2] >= cw - tol) & (x2 < W)) | ((b[:, 3] >= ch - tol) & (y2 < H)))
            b, s, c = b[~cut], s[~cut], c[~cut]
        if len(b):
            boxes.append(b + [x1, y1, x1, y1])
            scores.append(s)
            cls.append(c)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(cls)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def merge_detections(parts: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], iou_thr: float = 0.5,
                     class_agnostic: bool = False, return_source: bool = False):
    """Union of detection sets, then greedy NMS (highest score kept). return_source: also return, for every kept
    box, the index of the part it came from (e.g. 0 = full frame, 1 = zoom crops)."""
    src_all = np.concatenate([np.full(len(p[0]), i, dtype=int) for i, p in enumerate(parts)]) if parts else np.zeros(0, int)
    keep_parts = [len(p[0]) > 0 for p in parts]
    parts = [p for p, k in zip(parts, keep_parts) if k]
    if not parts:
        empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
        return (*empty, np.zeros(0, dtype=int)) if return_source else empty
    b = np.concatenate([np.asarray(p[0], float).reshape(-1, 4) for p in parts])
    s = np.concatenate([np.asarray(p[1], float).reshape(-1) for p in parts])
    c = np.concatenate([np.asarray(p[2]).reshape(-1).astype(int) for p in parts])
    order = np.argsort(-s, kind="stable")
    b, s, c, src_all = b[order], s[order], c[order], src_all[order]
    keep, remaining = [], np.arange(len(s))
    while remaining.size:   # greedy NMS, O(n * kept) memory-light
        i, rest = remaining[0], remaining[1:]
        keep.append(i)
        if not rest.size:
            break
        over = _iou(b[i:i + 1], b[rest])[0] > iou_thr
        if not class_agnostic:
            over &= c[rest] == c[i]
        remaining = rest[~over]
    keep = np.array(keep, dtype=int)
    if return_source:
        return b[keep], s[keep], c[keep], src_all[keep]
    return b[keep], s[keep], c[keep]

"""Optional — model-assisted annotation-completeness probe (indicative only, NOT ground truth).

Runs the project's own detectors (RT-DETR-L and YOLO26s) on
  (a) every image whose label file is empty, and
  (b) a random sample of labelled images,
and counts confident detections that no ground-truth box explains. Two-model agreement is required for
the "consensus" count, which is the number the report leans on. Caveat: the detectors were trained on the
train split (including its empty files as negatives), so train-split numbers are biased low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import PROJECT_ROOT


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _predict(model, paths: list[str], device: str, conf: float, batch: int = 16) -> list[tuple]:
    out = []
    for s in range(0, len(paths), batch):
        for r in model.predict(paths[s:s + batch], imgsz=640, conf=conf, device=device, verbose=False, save=False):
            b = r.boxes
            out.append((b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)))
    return out


def run_probe(targets: pd.DataFrame, gt_boxes: dict, device: str, conf_hi: float = 0.5, logger=None) -> pd.DataFrame:
    """targets: rows with abspath, relpath, split, prefix_class, empty_label. gt_boxes: relpath -> (N,4) xyxy px."""
    from ultralytics import RTDETR, YOLO

    paths = targets.abspath.tolist()
    if logger:
        logger.info(f"  model probe on {len(paths)} images (RT-DETR-L + YOLO26s, conf >= {conf_hi})")
    det_r = _predict(RTDETR(str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt")), paths, device, 0.25)
    det_s = _predict(YOLO(str(PROJECT_ROOT / "weights" / "yolo26s_best.pt")), paths, device, 0.25)
    rows = []
    for t, (br, sr, cr), (bs, ss, cs) in zip(targets.itertuples(), det_r, det_s):
        gt = gt_boxes.get(t.relpath, np.zeros((0, 4)))
        hr, hs = sr >= conf_hi, ss >= conf_hi
        br, cr, sr = br[hr], cr[hr], sr[hr]
        bs, cs, ss = bs[hs], cs[hs], ss[hs]
        # detections not explained by any GT box (class-agnostic IoU < 0.3)
        un_r = (_iou_matrix(br, gt).max(axis=1) < 0.3) if len(gt) else np.ones(len(br), bool)
        un_s = (_iou_matrix(bs, gt).max(axis=1) < 0.3) if len(gt) else np.ones(len(bs), bool)
        # consensus: an unexplained RT-DETR box that YOLO26s also predicts (same class, IoU >= 0.5)
        iou_rs = _iou_matrix(br, bs)
        cons = [i for i in range(len(br)) if un_r[i] and len(bs) and
                any(iou_rs[i, j] >= 0.5 and cr[i] == cs[j] and un_s[j] for j in range(len(bs)))]
        top = int(np.argmax(sr)) if len(sr) else -1
        rows.append({
            "relpath": t.relpath, "split": t.split, "prefix_class": t.prefix_class, "empty_label": t.empty_label,
            "n_gt": int(len(gt)),
            "n_rtdetr_hi": int(len(br)), "n_yolo26s_hi": int(len(bs)),
            "n_rtdetr_unexplained": int(un_r.sum()), "n_consensus_unexplained": len(cons),
            "consensus_classes": ",".join(str(int(cr[i])) for i in cons),
            "max_conf_rtdetr": float(sr.max()) if len(sr) else 0.0,
            "top_cls_rtdetr": int(cr[top]) if top >= 0 else -1,
            "consensus_boxes": [br[i].round(1).tolist() for i in cons],
        })
    return pd.DataFrame(rows)

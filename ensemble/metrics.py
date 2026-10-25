"""
Detection metrics for SWORD v2 — the single implementation every evaluation script should use.

Conventions (each one fixes a problem found in the 2026-10-22 dataset/evaluation audit):
    * mAP is computed from ALL predictions (confidence >= 0.001). Predictions are never truncated at a
      deployment threshold before computing AP — truncation caps recall and makes AP depend on the cut-off.
    * Precision / recall / F1 are reported (a) at the Ultralytics max-F1 point and (b) at an explicit
      operating threshold chosen on a separate tuning split, together with false positives per image.
    * COCO-style AP / AP_small / AP_medium / AP_large come from an independent COCO evaluator
      (faster-coco-eval, pycocotools-compatible), which handles size ranges with COCO's ignore rules.
    * Uncertainty: paired image-level bootstrap confidence intervals.

Per-image records (dicts) are the common currency:
    {"image_id": int, "path": str, "width": int, "height": int,
     "gt_cls": (G,) int, "gt_boxes": (G, 4) xyxy px,
     "pred_cls": (P,) int, "pred_scores": (P,) float, "pred_boxes": (P, 4) xyxy px}
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


# ───────────────────────────── ground truth ─────────────────────────────
def label_path_for(image_path: str) -> str:
    """Ultralytics convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    head, tail = image_path.rsplit(os.sep + "images" + os.sep, 1)
    return os.path.join(head, "labels", os.path.splitext(tail)[0] + ".txt")


def load_yolo_labels(label_path: str, img_w: int, img_h: int):
    """YOLO txt -> (classes (G,), boxes (G, 4) xyxy in pixels). Missing or empty file -> no objects."""
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    data = np.loadtxt(label_path, ndmin=2)
    if data.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 4), dtype=np.float64)
    cls = data[:, 0].astype(np.int64)
    cx, cy, w, h = data[:, 1] * img_w, data[:, 2] * img_h, data[:, 3] * img_w, data[:, 4] * img_h
    return cls, np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


# ───────────────────────────── matching ─────────────────────────────
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_image(rec: dict) -> np.ndarray:
    """(P, 10) boolean TP matrix over IoU 0.50:0.95, Ultralytics matching (largest-IoU-first, one-to-one, same class)."""
    p_cls, g_cls = rec["pred_cls"], rec["gt_cls"]
    tp = np.zeros((len(p_cls), len(IOU_THRESHOLDS)), dtype=bool)
    if len(p_cls) == 0 or len(g_cls) == 0:
        return tp
    iou = box_iou(rec["gt_boxes"], rec["pred_boxes"]) * (g_cls[:, None] == p_cls[None, :])
    for t, thr in enumerate(IOU_THRESHOLDS):
        gi, pi = np.nonzero(iou >= thr)
        if len(gi) == 0:
            continue
        order = np.argsort(-iou[gi, pi], kind="stable")
        gi, pi = gi[order], pi[order]
        _, first_p = np.unique(pi, return_index=True)
        gi, pi = gi[first_p], pi[first_p]
        _, first_g = np.unique(gi, return_index=True)
        tp[pi[first_g], t] = True
    return tp


def precompute_tp(records: Sequence[dict]) -> None:
    """Cache each record's TP matrix (key "_tp") so repeated metric calls (bootstrap) skip re-matching."""
    for r in records:
        r["_tp"] = match_image(r)


# ───────────────────────────── AP (Ultralytics convention) ─────────────────────────────
def detection_metrics(records: Sequence[dict], names: Sequence[str]) -> dict:
    """mAP50, mAP50-95, per-class AP and P/R/F1 at the max-F1 point, from ALL predictions."""
    from ultralytics.utils.metrics import ap_per_class

    tps, confs, pcls, tcls = [], [], [], []
    for r in records:
        tcls.append(r["gt_cls"])
        if len(r["pred_cls"]):
            tps.append(r["_tp"] if "_tp" in r else match_image(r))
            confs.append(r["pred_scores"])
            pcls.append(r["pred_cls"])
    target = np.concatenate(tcls) if tcls else np.zeros(0)
    if not tps or len(target) == 0:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "per_class": {},
                "n_images": len(records), "n_gt": int(len(target)), "n_pred": 0}
    tp, fp, p, r, f1, ap, classes, *_ = ap_per_class(
        np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), target,
        plot=False, names=dict(enumerate(names)))
    per_class = {names[int(c)]: {"ap50": float(ap[i, 0]), "ap50_95": float(ap[i].mean()), "precision": float(p[i]),
                                 "recall": float(r[i]), "f1": float(f1[i]), "n_gt": int((target == c).sum())}
                 for i, c in enumerate(classes)}
    return {"mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()), "precision": float(p.mean()),
            "recall": float(r.mean()), "f1": float(f1.mean()), "per_class": per_class,
            "n_images": len(records), "n_gt": int(len(target)), "n_pred": int(sum(len(c) for c in confs))}


# ───────────────────────────── operating point ─────────────────────────────
def operating_point(records: Sequence[dict], conf_thr: float, n_classes: int, iou_thr: float = 0.5) -> dict:
    """Precision / recall / F1 when only detections with score >= conf_thr are kept (what a deployed system sees)."""
    t = int(np.argmin(np.abs(IOU_THRESHOLDS - iou_thr)))
    tp_c, fp_c, fn_c = np.zeros(n_classes), np.zeros(n_classes), np.zeros(n_classes)
    fp_images = fp_on_empty = n_empty = 0
    for r in records:
        keep = r["pred_scores"] >= conf_thr
        sub = {**r, "pred_cls": r["pred_cls"][keep], "pred_scores": r["pred_scores"][keep], "pred_boxes": r["pred_boxes"][keep]}
        tp = match_image(sub)[:, t] if len(sub["pred_cls"]) else np.zeros(0, dtype=bool)
        for c in range(n_classes):
            pc = sub["pred_cls"] == c
            tp_c[c] += tp[pc].sum()
            fp_c[c] += (~tp[pc]).sum()
            fn_c[c] += (r["gt_cls"] == c).sum() - tp[pc].sum()
        n_fp = int((~tp).sum())
        fp_images += n_fp > 0
        if len(r["gt_cls"]) == 0:
            n_empty += 1
            fp_on_empty += n_fp
    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    rec = tp_c / np.maximum(tp_c + fn_c, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = (tp_c + fn_c) > 0
    return {"conf_thr": conf_thr, "iou_thr": iou_thr,
            "precision": float(prec[present].mean()) if present.any() else 0.0,
            "recall": float(rec[present].mean()) if present.any() else 0.0,
            "f1": float(f1[present].mean()) if present.any() else 0.0,
            "micro_precision": float(tp_c.sum() / max(tp_c.sum() + fp_c.sum(), 1)),
            "micro_recall": float(tp_c.sum() / max(tp_c.sum() + fn_c.sum(), 1)),
            "false_positives": int(fp_c.sum()), "fp_per_image": float(fp_c.sum() / max(len(records), 1)),
            "images_with_any_fp": int(fp_images),
            "images_without_gt": n_empty, "fp_per_image_without_gt": float(fp_on_empty / max(n_empty, 1))}


def best_threshold(records: Sequence[dict], n_classes: int, grid: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))) -> float:
    """Threshold maximising macro F1 — choose it on the TUNING split, then apply it unchanged to the test split."""
    scores = [(operating_point(records, float(t), n_classes)["f1"], float(t)) for t in grid]
    return max(scores)[1]


# ───────────────────────────── COCO evaluator ─────────────────────────────
def coco_metrics(records: Sequence[dict], names: Sequence[str]) -> dict:
    """AP, AP50, AP75, AP_small/medium/large, AR@100 from faster-coco-eval (falls back to pycocotools)."""
    try:
        from faster_coco_eval import COCO, COCOeval_faster as COCOeval
    except ImportError:  # pragma: no cover
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    images, anns, dets = [], [], []
    aid = 1
    for r in records:
        images.append({"id": int(r["image_id"]), "width": int(r["width"]), "height": int(r["height"])})
        for c, b in zip(r["gt_cls"], r["gt_boxes"]):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": aid, "image_id": int(r["image_id"]), "category_id": int(c), "bbox": [float(b[0]), float(b[1]), w, h],
                         "area": w * h, "iscrowd": 0})
            aid += 1
        for c, s, b in zip(r["pred_cls"], r["pred_scores"], r["pred_boxes"]):
            dets.append({"image_id": int(r["image_id"]), "category_id": int(c), "score": float(s),
                         "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": i, "name": n} for i, n in enumerate(names)]}
    gt.createIndex()
    if not dets:
        return {"AP": 0.0}
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    stats = [float(x) for x in ev.stats[:12]]
    out = dict(zip(keys, stats))
    # COCO reports -1 when a size range has no ground truth: make that explicit instead of a number
    for k in list(out):
        if out[k] < 0:
            out[k] = None
    areas = np.array([a["area"] for a in anns]) if anns else np.zeros(0)
    out["n_gt_small"] = int((areas < 32 ** 2).sum())
    out["n_gt_medium"] = int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum())
    out["n_gt_large"] = int((areas >= 96 ** 2).sum())
    return out


# ───────────────────────────── uncertainty ─────────────────────────────
def bootstrap_ci(records: Sequence[dict], metric: Callable[[List[dict]], float], n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05, paired_with: Optional[Sequence[dict]] = None) -> dict:
    """Image-level bootstrap. With paired_with, returns the CI of metric(records) - metric(paired_with)
    using the same resampled images for both (records and paired_with must list the same images in order)."""
    rng = np.random.default_rng(seed)
    n = len(records)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = metric([records[i] for i in idx])
        vals.append(a - metric([paired_with[i] for i in idx]) if paired_with is not None else a)
    vals = np.asarray(vals)
    point = metric(list(records)) - (metric(list(paired_with)) if paired_with is not None else 0.0)
    return {"point": float(point), "low": float(np.quantile(vals, alpha / 2)), "high": float(np.quantile(vals, 1 - alpha / 2)),
            "n_boot": n_boot, "p_le_zero": float((vals <= 0).mean()) if paired_with is not None else None}


def fast_map(names: Sequence[str]) -> Dict[str, Callable[[List[dict]], float]]:
    """Metric callables for bootstrapping."""
    return {"mAP50": lambda rs: detection_metrics(rs, names)["mAP50"],
            "mAP50-95": lambda rs: detection_metrics(rs, names)["mAP50-95"]}

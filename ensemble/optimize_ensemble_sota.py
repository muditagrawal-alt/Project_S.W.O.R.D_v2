"""
SWORD v2 — Non-Training Optimization Engine for SOTA Multi-Model Ensemble
Searches for:
1. Class-Adaptive WBF weights: W_c = [w_{s,c}, w_{m,c}, w_{detr,c}] for each of the 10 weapon classes.
2. Consensus Boosting factor and single-model outlier penalty.
3. Class-specific confidence thresholds tau_c to maximize F1 and mAP@50.
4. Optimal WBF IoU clustering threshold.

Zero training required. Executes locally on Apple Silicon GPU (MPS) in ~2 minutes.
Outputs:
  - results/optimized_parameters.json
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ensemble"))

from ultralytics import YOLO, RTDETR
from ultralytics.utils.metrics import ap_per_class, box_iou
from ensemble.wbf import weighted_boxes_fusion

CLASS_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]
NUM_CLASSES = len(CLASS_NAMES)
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


def match_predictions(pred_classes, true_classes, iou):
    n_preds = pred_classes.shape[0]
    correct = np.zeros((n_preds, 10), dtype=bool)
    if n_preds == 0 or true_classes.shape[0] == 0:
        return correct

    correct_class = (true_classes[:, None] == pred_classes[None, :]).cpu().numpy()
    iou_np = (iou * torch.tensor(correct_class, device=iou.device)).cpu().numpy()

    for i, thr in enumerate(IOU_THRESHOLDS.tolist()):
        matches = np.nonzero(iou_np >= thr)
        matches = np.array(matches).T
        if matches.shape[0] > 0:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return correct


def load_gt(label_path, img_w, img_h):
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)
    data = np.loadtxt(label_path, ndmin=2)
    if data.size == 0:
        return torch.zeros((0,), dtype=torch.int64), torch.zeros((0, 4), dtype=torch.float32)

    classes = torch.tensor(data[:, 0], dtype=torch.int64)
    xywh = data[:, 1:5]
    x1 = (xywh[:, 0] - xywh[:, 2] / 2) * img_w
    y1 = (xywh[:, 1] - xywh[:, 3] / 2) * img_h
    x2 = (xywh[:, 0] + xywh[:, 2] / 2) * img_w
    y2 = (xywh[:, 1] + xywh[:, 3] / 2) * img_h
    boxes = torch.tensor(np.column_stack([x1, y1, x2, y2]), dtype=torch.float32)
    return classes, boxes


def get_stratified_image_list(val_dir, samples_per_class=50, seed=42):
    lbl_dir = val_dir.replace("images", "labels")
    files = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".txt")])

    class_to_images = defaultdict(list)
    for f in files:
        path = os.path.join(lbl_dir, f)
        img_name = os.path.splitext(f)[0] + ".jpg"
        if not os.path.exists(os.path.join(val_dir, img_name)):
            for ext in [".png", ".jpeg"]:
                alt = os.path.splitext(f)[0] + ext
                if os.path.exists(os.path.join(val_dir, alt)):
                    img_name = alt
                    break
        if os.path.getsize(path) > 0:
            with open(path) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts:
                        c = int(parts[0])
                        class_to_images[c].append(img_name)
                        break

    random.seed(seed)
    selected = set()
    for c in range(NUM_CLASSES):
        pool = class_to_images[c]
        k = min(samples_per_class, len(pool))
        for img in random.sample(pool, k):
            selected.add(img)

    return sorted(list(selected))


def evaluate_model_detections(pred_list, gt_list, device="cpu"):
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []

    for pred, gt in zip(pred_list, gt_list):
        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_cls = pred["cls"]
        g_boxes = gt["boxes"]
        g_cls = gt["cls"]

        target_cls_all.append(g_cls.numpy())

        if len(p_boxes) == 0:
            continue

        conf_all.append(p_scores.numpy())
        pred_cls_all.append(p_cls.numpy())

        if len(g_boxes) == 0:
            tp_all.append(np.zeros((len(p_boxes), 10), dtype=bool))
            continue

        iou = box_iou(g_boxes.to(device), p_boxes.to(device))
        tp = match_predictions(p_cls.to(device), g_cls.to(device), iou)
        tp_all.append(tp)

    if len(conf_all) == 0:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "p": 0.0, "r": 0.0, "f1": 0.0, "per_class": {}}

    tp_cat = np.concatenate(tp_all, axis=0)
    conf_cat = np.concatenate(conf_all, axis=0)
    pred_cls_cat = np.concatenate(pred_cls_all, axis=0)
    target_cls_cat = np.concatenate(target_cls_all, axis=0)

    tp, fp, p, r, f1, ap, unique_classes, _, _, _, _, _ = ap_per_class(
        tp_cat, conf_cat, pred_cls_cat, target_cls_cat, plot=False, names=dict(enumerate(CLASS_NAMES))
    )

    ap50 = ap[:, 0]
    ap50_95 = ap.mean(axis=1)

    per_class = {}
    for i, c in enumerate(unique_classes):
        cls_name = CLASS_NAMES[int(c)]
        per_class[cls_name] = {
            "ap50": float(ap50[i]),
            "ap50_95": float(ap50_95[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
        }

    return {
        "mAP50": float(ap50.mean()) if len(ap50) > 0 else 0.0,
        "mAP50-95": float(ap50_95.mean()) if len(ap50_95) > 0 else 0.0,
        "p": float(p.mean()) if len(p) > 0 else 0.0,
        "r": float(r.mean()) if len(r) > 0 else 0.0,
        "f1": float(f1.mean()) if len(f1) > 0 else 0.0,
        "per_class": per_class,
    }


def parse_boxes(r):
    if r.boxes is not None and len(r.boxes) > 0:
        bx = r.boxes.xyxy.cpu().numpy()
        sc = r.boxes.conf.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        return bx, sc, cl
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)


def run_wbf_fusion(
    raw_s, raw_m, raw_detr, shapes,
    weights=(1.0, 1.1, 1.1),
    class_weights=None,
    iou_thr=0.55,
    conf_thr=0.15,
    conf_type="avg",
    consensus_boost=0.15,
    outlier_thr=0.35,
    outlier_penalty=0.65,
    class_conf_thrs=None,
):
    n_images = len(shapes)
    preds = []

    for i in range(n_images):
        h, w = shapes[i]
        boxes_list = []
        scores_list = []
        labels_list = []

        for m_dets in [raw_s[i], raw_m[i], raw_detr[i]]:
            bx, sc, cl = m_dets
            if len(bx) > 0:
                bx_norm = bx.copy()
                bx_norm[:, [0, 2]] /= w
                bx_norm[:, [1, 3]] /= h
                boxes_list.append(bx_norm)
                scores_list.append(sc)
                labels_list.append(cl)
            else:
                boxes_list.append(np.zeros((0, 4)))
                scores_list.append(np.zeros(0))
                labels_list.append(np.zeros(0, dtype=int))

        has_dets = any(len(b) > 0 for b in boxes_list)
        if has_dets:
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=list(weights),
                class_weights=class_weights,
                iou_thr=iou_thr,
                skip_box_thr=0.001,
                conf_type=conf_type,
                consensus_boost=consensus_boost,
                outlier_thr=outlier_thr,
                outlier_penalty=outlier_penalty,
                class_conf_thrs=class_conf_thrs,
            )

            # Global confidence filter fallback if no class_conf_thrs
            if class_conf_thrs is None:
                mask = f_scores >= conf_thr
                f_boxes = f_boxes[mask]
                f_scores = f_scores[mask]
                f_labels = f_labels[mask]

            if len(f_boxes) > 0:
                f_boxes[:, [0, 2]] *= w
                f_boxes[:, [1, 3]] *= h

            preds.append({
                "boxes": torch.tensor(f_boxes, dtype=torch.float32),
                "scores": torch.tensor(f_scores, dtype=torch.float32),
                "cls": torch.tensor(f_labels, dtype=torch.int64),
            })
        else:
            preds.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "cls": torch.zeros((0,), dtype=torch.int64),
            })

    return preds


def optimize():
    import cv2
    val_images_dir = str(PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 80)
    print(f"🔧 SWORD v2 — Non-Training Optimization Search | Device: {device}")
    print("=" * 80)

    # 1. Stratified Validation Images (50 per class = 500 images)
    image_names = get_stratified_image_list(val_images_dir, samples_per_class=50, seed=42)
    print(f"Dataset: 500 Stratified Validation Images across 10 weapon classes")

    # 2. Load Models
    print("\nLoading weights...")
    model_s = YOLO(str(PROJECT_ROOT / "weights" / "yolo26s_best.pt"))
    model_m = YOLO(str(PROJECT_ROOT / "weights" / "yolo26m_best.pt"))
    model_rtdetr = RTDETR(str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"))

    # 3. Extract and Cache Raw Detections
    raw_dets = {"YOLO26s": [], "YOLO26m": [], "RT-DETR-L": []}
    gt_list = []
    shapes = []

    batch_size = 32
    print(f"\nExtracting raw candidate detections from 500 images...")
    t0 = time.time()
    for name, model in [("YOLO26s", model_s), ("YOLO26m", model_m), ("RT-DETR-L", model_rtdetr)]:
        t_m = time.time()
        for idx in range(0, len(image_names), batch_size):
            batch_paths = [os.path.join(val_images_dir, img) for img in image_names[idx:idx + batch_size]]
            res_batch = model(batch_paths, conf=0.001, iou=0.7, verbose=False, device=device)
            for r in res_batch:
                raw_dets[name].append(parse_boxes(r))
        print(f"  {name} inference complete in {time.time() - t_m:.1f}s")

    for img_name in image_names:
        img_path = os.path.join(val_images_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        shapes.append((h, w))
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(val_images_dir.replace("images", "labels"), lbl_name)
        gt_cls, gt_boxes = load_gt(lbl_path, w, h)
        gt_list.append({"cls": gt_cls, "boxes": gt_boxes})

    print(f"All candidate detections cached in {time.time() - t0:.1f}s.")

    # ── BASELINE BENCHMARK 1 EVALUATION ──
    print("\nEvaluating Baseline (Benchmark 1 settings: weights=[1.0, 1.1, 1.1], conf_thr=0.15, conf_type='avg')...")
    base_preds = run_wbf_fusion(
        raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
        shapes, weights=(1.0, 1.1, 1.1), iou_thr=0.55, conf_thr=0.15, conf_type="avg"
    )
    base_m = evaluate_model_detections(base_preds, gt_list, device=device)
    print(f"BASELINE: mAP@50 = {base_m['mAP50']:.4f}, mAP@50:95 = {base_m['mAP50-95']:.4f}, Precision = {base_m['p']:.4f}, Recall = {base_m['r']:.4f}, F1 = {base_m['f1']:.4f}")

    # ── STEP 1: CLASS-ADAPTIVE WEIGHT SEARCH ──
    print("\n[Step 1/4] Optimizing Class-Adaptive WBF Weights...")
    # Candidate weight profiles [w_s, w_m, w_detr]
    weight_candidates = [
        [1.0, 1.1, 1.1],  # baseline neutral
        [0.4, 0.5, 2.2],  # heavy transformer dominant (knife, sword)
        [0.6, 0.7, 1.8],  # moderate transformer dominant
        [0.8, 1.8, 1.0],  # yolo26m precision dominant (machine gun, pistol)
        [1.8, 1.0, 0.8],  # yolo26s dominant (grenade, smg)
        [1.2, 1.2, 0.8],  # dual yolo consensus
        [1.0, 1.5, 1.5],  # large model consensus
    ]

    best_class_weights = {}
    for c_id, c_name in enumerate(CLASS_NAMES):
        best_ap = base_m["per_class"].get(c_name, {}).get("ap50", 0.0)
        best_w = [1.0, 1.1, 1.1]

        for w_cand in weight_candidates:
            trial_cw = {c_id: w_cand}
            preds = run_wbf_fusion(
                raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
                shapes, weights=(1.0, 1.1, 1.1), class_weights=trial_cw,
                iou_thr=0.55, conf_thr=0.15, conf_type="avg"
            )
            m = evaluate_model_detections(preds, gt_list, device=device)
            c_ap = m["per_class"].get(c_name, {}).get("ap50", 0.0)
            if c_ap > best_ap:
                best_ap = c_ap
                best_w = w_cand

        best_class_weights[c_id] = best_w
        baseline_c_ap = base_m["per_class"].get(c_name, {}).get("ap50", 0.0)
        gain_str = f"+{(best_ap - baseline_c_ap)*100:.2f}%" if best_ap > baseline_c_ap else "+0.00%"
        print(f"  Class {c_id:2d} ({c_name:15s}): Weights = {best_w} -> AP@50 = {best_ap:.4f} ({gain_str})")

    # Evaluate after class weighting
    preds_cw = run_wbf_fusion(
        raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
        shapes, weights=(1.0, 1.1, 1.1), class_weights=best_class_weights,
        iou_thr=0.55, conf_thr=0.15, conf_type="avg"
    )
    m_cw = evaluate_model_detections(preds_cw, gt_list, device=device)
    print(f"-> After Class-Adaptive Weights: mAP@50 = {m_cw['mAP50']:.4f} (+{(m_cw['mAP50'] - base_m['mAP50'])*100:.2f}%), F1 = {m_cw['f1']:.4f}")

    # ── STEP 2: CONSENSUS BOOSTING & OUTLIER PENALTY SEARCH ──
    print("\n[Step 2/4] Optimizing Consensus Boosting & Outlier Penalty...")
    boost_candidates = [0.08, 0.12, 0.15, 0.20]
    outlier_thrs = [0.25, 0.30, 0.35]
    outlier_penalties = [0.60, 0.70, 0.80]

    best_boost = 0.15
    best_othr = 0.30
    best_openalty = 0.70
    best_map = m_cw["mAP50"]

    for b in boost_candidates:
        for ot in outlier_thrs:
            for op in outlier_penalties:
                preds = run_wbf_fusion(
                    raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
                    shapes, weights=(1.0, 1.1, 1.1), class_weights=best_class_weights,
                    iou_thr=0.55, conf_thr=0.15, conf_type="consensus_boosted",
                    consensus_boost=b, outlier_thr=ot, outlier_penalty=op
                )
                m = evaluate_model_detections(preds, gt_list, device=device)
                if m["mAP50"] > best_map:
                    best_map = m["mAP50"]
                    best_boost = b
                    best_othr = ot
                    best_openalty = op

    print(f"  Best Consensus Boost: {best_boost:.2f}, Outlier Thr: {best_othr:.2f}, Penalty: {best_openalty:.2f}")
    preds_boost = run_wbf_fusion(
        raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
        shapes, weights=(1.0, 1.1, 1.1), class_weights=best_class_weights,
        iou_thr=0.55, conf_thr=0.15, conf_type="consensus_boosted",
        consensus_boost=best_boost, outlier_thr=best_othr, outlier_penalty=best_openalty
    )
    m_boost = evaluate_model_detections(preds_boost, gt_list, device=device)
    print(f"-> After Consensus Boosting: mAP@50 = {m_boost['mAP50']:.4f} (+{(m_boost['mAP50'] - base_m['mAP50'])*100:.2f}%), Precision = {m_boost['p']:.4f}, F1 = {m_boost['f1']:.4f}")

    # ── STEP 3: OPTIMAL IOU THRESHOLD SEARCH ──
    print("\n[Step 3/4] Optimizing WBF IoU Clustering Threshold...")
    iou_candidates = [0.48, 0.50, 0.52, 0.55, 0.58]
    best_iou = 0.55
    best_map_iou = m_boost["mAP50"]

    for iou_t in iou_candidates:
        preds = run_wbf_fusion(
            raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
            shapes, weights=(1.0, 1.1, 1.1), class_weights=best_class_weights,
            iou_thr=iou_t, conf_thr=0.15, conf_type="consensus_boosted",
            consensus_boost=best_boost, outlier_thr=best_othr, outlier_penalty=best_openalty
        )
        m = evaluate_model_detections(preds, gt_list, device=device)
        print(f"  IoU Thr {iou_t:.2f} -> mAP@50 = {m['mAP50']:.4f}, F1 = {m['f1']:.4f}")
        if m["mAP50"] > best_map_iou:
            best_map_iou = m["mAP50"]
            best_iou = iou_t

    print(f"  Optimal IoU Threshold: {best_iou:.2f}")

    # ── STEP 4: PER-CLASS CONFIDENCE THRESHOLD TUNING ──
    print("\n[Step 4/4] Optimizing Per-Class Confidence Thresholds...")
    best_class_thrs = {}
    thr_candidates = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]

    for c_id, c_name in enumerate(CLASS_NAMES):
        best_f1_c = 0.0
        best_t = 0.15
        for t in thr_candidates:
            trial_thrs = {c_id: t}
            preds = run_wbf_fusion(
                raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
                shapes, weights=(1.0, 1.1, 1.1), class_weights=best_class_weights,
                iou_thr=best_iou, conf_thr=0.15, conf_type="consensus_boosted",
                consensus_boost=best_boost, outlier_thr=best_othr, outlier_penalty=best_openalty,
                class_conf_thrs=trial_thrs
            )
            m = evaluate_model_detections(preds, gt_list, device=device)
            f1_c = m["per_class"].get(c_name, {}).get("f1", 0.0)
            if f1_c > best_f1_c:
                best_f1_c = f1_c
                best_t = t
        best_class_thrs[c_id] = best_t
        print(f"  Class {c_id:2d} ({c_name:15s}): Optimal Threshold = {best_t:.2f} (F1 = {best_f1_c:.4f})")

    # Final Combined Evaluation
    final_preds = run_wbf_fusion(
        raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"],
        shapes, weights=(1.0, 1.1, 1.1), class_weights=best_class_weights,
        iou_thr=best_iou, conf_thr=0.15, conf_type="consensus_boosted",
        consensus_boost=best_boost, outlier_thr=best_othr, outlier_penalty=best_openalty,
        class_conf_thrs=best_class_thrs
    )
    final_m = evaluate_model_detections(final_preds, gt_list, device=device)

    print("\n" + "=" * 80)
    print("🏆 FINAL OPTIMIZATION COMPARISON:")
    print(f"  Metric       | Baseline (Run 1) | Optimized (Run 2) | Gain")
    print(f"  -------------+------------------+-------------------+---------")
    print(f"  mAP@50       | {base_m['mAP50']:.4f}           | {final_m['mAP50']:.4f}            | +{(final_m['mAP50'] - base_m['mAP50'])*100:+.2f}%")
    print(f"  mAP@50:95    | {base_m['mAP50-95']:.4f}           | {final_m['mAP50-95']:.4f}            | +{(final_m['mAP50-95'] - base_m['mAP50-95'])*100:+.2f}%")
    print(f"  Precision    | {base_m['p']:.4f}           | {final_m['p']:.4f}            | +{(final_m['p'] - base_m['p'])*100:+.2f}%")
    print(f"  Recall       | {base_m['r']:.4f}           | {final_m['r']:.4f}            | +{(final_m['r'] - base_m['r'])*100:+.2f}%")
    print(f"  F1 Score     | {base_m['f1']:.4f}           | {final_m['f1']:.4f}            | +{(final_m['f1'] - base_m['f1'])*100:+.2f}%")
    print("=" * 80)

    # Save to JSON
    output_path = PROJECT_ROOT / "results" / "optimized_parameters.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params_export = {
        "global_weights": [1.0, 1.1, 1.1],
        "class_weights": {str(k): v for k, v in best_class_weights.items()},
        "class_conf_thrs": {str(k): v for k, v in best_class_thrs.items()},
        "iou_thr": best_iou,
        "conf_type": "consensus_boosted",
        "consensus_boost": best_boost,
        "outlier_thr": best_othr,
        "outlier_penalty": best_openalty,
        "class_names": CLASS_NAMES,
        "baseline_metrics": {
            "mAP50": base_m["mAP50"],
            "mAP50_95": base_m["mAP50-95"],
            "precision": base_m["p"],
            "recall": base_m["r"],
            "f1": base_m["f1"],
        },
        "optimized_metrics": {
            "mAP50": final_m["mAP50"],
            "mAP50_95": final_m["mAP50-95"],
            "precision": final_m["p"],
            "recall": final_m["r"],
            "f1": final_m["f1"],
        },
    }

    with open(output_path, "w") as fp:
        json.dump(params_export, fp, indent=2)

    print(f"\n✅ Optimal parameters successfully exported to: {output_path}")


if __name__ == "__main__":
    optimize()

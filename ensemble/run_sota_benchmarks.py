"""
SWORD v2 — Industry SOTA Benchmark & Research Paper Evaluation Suite
Computes standardized academic and commercial surveillance benchmarks:
1. Macro & Per-Class Metrics across all 10 weapon classes.
2. SO-WDet Object Scale Breakdown: Small (<32^2), Medium, Large objects (AP_s, AP_m, AP_l).
3. False Alarm Rate (FAR) & Confidence Calibration Sweep (Commercial Security Standard).
4. WBF IoU Threshold Sensitivity Ablation Study.
5. Automated generation of publication-ready LaTeX tables and high-resolution figures.

Usage:
    python ensemble/run_sota_benchmarks.py --samples-per-class 50 --device mps
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
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Set high-quality styling for research paper plots
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def match_predictions(pred_classes, true_classes, iou):
    """Match predictions to ground truths across multiple IoU thresholds."""
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
    """Load ground truth labels from YOLO format .txt file."""
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
    """Get a balanced sample of images across all 10 classes."""
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
    """Compute mAP, precision, recall, and F1 using Ultralytics metrics."""
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


def compute_scale_metrics(pred_list, gt_list, device="cpu"):
    """
    Compute AP for Small (< 32^2), Medium (32^2 - 96^2), and Large (>= 96^2) objects.
    Standard COCO / SO-WDet object scale segmentation.
    """
    scale_definitions = {
        "Small (< 32px)": (0, 32**2),
        "Medium (32-96px)": (32**2, 96**2),
        "Large (> 96px)": (96**2, float("inf")),
    }
    
    scale_results = {}
    for scale_name, (min_area, max_area) in scale_definitions.items():
        scaled_preds = []
        scaled_gts = []
        
        for p, g in zip(pred_list, gt_list):
            # Filter GT by area
            if len(g["boxes"]) > 0:
                g_areas = (g["boxes"][:, 2] - g["boxes"][:, 0]) * (g["boxes"][:, 3] - g["boxes"][:, 1])
                g_mask = (g_areas >= min_area) & (g_areas < max_area)
                s_g = {"boxes": g["boxes"][g_mask], "cls": g["cls"][g_mask]}
            else:
                s_g = {"boxes": g["boxes"], "cls": g["cls"]}
                
            # Filter Pred by area
            if len(p["boxes"]) > 0:
                p_areas = (p["boxes"][:, 2] - p["boxes"][:, 0]) * (p["boxes"][:, 3] - p["boxes"][:, 1])
                p_mask = (p_areas >= min_area) & (p_areas < max_area)
                s_p = {
                    "boxes": p["boxes"][p_mask],
                    "scores": p["scores"][p_mask],
                    "cls": p["cls"][p_mask],
                }
            else:
                s_p = {"boxes": p["boxes"], "scores": p["scores"], "cls": p["cls"]}
                
            scaled_preds.append(s_p)
            scaled_gts.append(s_g)
            
        m = evaluate_model_detections(scaled_preds, scaled_gts, device=device)
        scale_results[scale_name] = m["mAP50"]
        
    return scale_results


def parse_boxes(r):
    """Extract boxes, scores, and classes from an Ultralytics result."""
    if r.boxes is not None and len(r.boxes) > 0:
        bx = r.boxes.xyxy.cpu().numpy()
        sc = r.boxes.conf.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        return bx, sc, cl
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)


def filter_preds(raw_dets, conf_thr=0.20):
    """Apply confidence threshold to raw detections."""
    preds = []
    for bx, sc, cl in raw_dets:
        if len(sc) > 0:
            mask = sc >= conf_thr
            preds.append({
                "boxes": torch.tensor(bx[mask], dtype=torch.float32),
                "scores": torch.tensor(sc[mask], dtype=torch.float32),
                "cls": torch.tensor(cl[mask], dtype=torch.int64),
            })
        else:
            preds.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "cls": torch.zeros((0,), dtype=torch.int64),
            })
    return preds


def fuse_wbf(model_dets_list, shapes, weights, iou_thr=0.55, conf_thr=0.20, conf_type="avg"):
    """Fuse raw detections across models using Weighted Boxes Fusion."""
    n_images = len(shapes)
    n_models = len(model_dets_list)
    preds = []

    for i in range(n_images):
        h, w = shapes[i]
        boxes_list = []
        scores_list = []
        labels_list = []

        for m in range(n_models):
            bx, sc, cl = model_dets_list[m][i]
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
                weights=weights, iou_thr=iou_thr, skip_box_thr=0.001,
                conf_type=conf_type,
            )
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


def run_full_sota_benchmark():
    import cv2
    val_images_dir = str(PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val")
    output_dir = PROJECT_ROOT / "results" / "paper_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 80)
    print(f"🚀 SWORD v2 — SOTA Industry & Academic Benchmark Suite | Device: {device}")
    print("=" * 80)

    # 1. Stratified Validation Images (50 per class = 500 images)
    image_names = get_stratified_image_list(val_images_dir, samples_per_class=50, seed=42)
    print(f"Dataset: 500 Stratified Validation Images across 10 weapon classes")

    # 2. Load Models
    print("\nLoading weights...")
    model_s = YOLO(str(PROJECT_ROOT / "weights" / "yolo26s_best.pt"))
    model_m = YOLO(str(PROJECT_ROOT / "weights" / "yolo26m_best.pt"))
    model_rtdetr = RTDETR(str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"))

    models = [model_s, model_m, model_rtdetr]
    model_names = ["YOLO26s", "YOLO26m", "RT-DETR-L"]

    # 3. Batch Inference & Latency Tracking
    raw_dets = {name: [] for name in model_names}
    gt_list = []
    shapes = []

    batch_size = 32
    print(f"\nRunning batch inference on {len(image_names)} images...")
    t_start = time.time()
    latencies = {}

    for name, model in zip(model_names, models):
        t_model = time.time()
        for idx in range(0, len(image_names), batch_size):
            batch_paths = [os.path.join(val_images_dir, img) for img in image_names[idx:idx + batch_size]]
            res_batch = model(batch_paths, conf=0.001, iou=0.7, verbose=False, device=device)
            for r in res_batch:
                raw_dets[name].append(parse_boxes(r))
        latencies[name] = (time.time() - t_model) / len(image_names) * 1000  # ms/img

    # Load Ground Truths
    for img_name in image_names:
        img_path = os.path.join(val_images_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        shapes.append((h, w))
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(val_images_dir.replace("images", "labels"), lbl_name)
        gt_cls, gt_boxes = load_gt(lbl_path, w, h)
        gt_list.append({"cls": gt_cls, "boxes": gt_boxes})

    total_infer_time = time.time() - t_start
    print(f"Inference complete in {total_infer_time:.1f}s")

    # 4. Standard Metric Evaluation across all configurations
    results = {}
    scale_metrics = {}

    # Standalone models
    for name in model_names:
        preds = filter_preds(raw_dets[name], conf_thr=0.20)
        results[name] = evaluate_model_detections(preds, gt_list, device=device)
        scale_metrics[name] = compute_scale_metrics(preds, gt_list, device=device)

    # 2-Model Ensemble
    t_wbf2 = time.time()
    preds_ens2 = fuse_wbf([raw_dets["YOLO26s"], raw_dets["YOLO26m"]], shapes, weights=[1.0, 1.1], conf_thr=0.20)
    latencies["Ensemble (2-Model)"] = latencies["YOLO26s"] + latencies["YOLO26m"] + ((time.time() - t_wbf2) / len(image_names) * 1000)
    results["Ensemble (2-Model)"] = evaluate_model_detections(preds_ens2, gt_list, device=device)
    scale_metrics["Ensemble (2-Model)"] = compute_scale_metrics(preds_ens2, gt_list, device=device)

    # 3-Model Ensemble
    t_wbf3 = time.time()
    preds_ens3 = fuse_wbf([raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"]], shapes, weights=[1.0, 1.1, 1.1], conf_thr=0.15, conf_type="avg")
    latencies["Ensemble (3-Model)"] = latencies["YOLO26s"] + latencies["YOLO26m"] + latencies["RT-DETR-L"] + ((time.time() - t_wbf3) / len(image_names) * 1000)
    results["Ensemble (3-Model)"] = evaluate_model_detections(preds_ens3, gt_list, device=device)
    scale_metrics["Ensemble (3-Model)"] = compute_scale_metrics(preds_ens3, gt_list, device=device)

    # 5. Benchmark 3: False Alarm Rate & Confidence Calibration Sweep
    print("\nEvaluating False Alarm Rate & Confidence Calibration...")
    conf_sweep = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    calibration_data = {cfg: {"conf": conf_sweep, "precision": [], "recall": [], "fdr": []} for cfg in results.keys()}

    for c in conf_sweep:
        for name in model_names:
            p = filter_preds(raw_dets[name], conf_thr=c)
            m = evaluate_model_detections(p, gt_list, device=device)
            calibration_data[name]["precision"].append(m["p"])
            calibration_data[name]["recall"].append(m["r"])
            calibration_data[name]["fdr"].append(1.0 - m["p"])

        p2 = fuse_wbf([raw_dets["YOLO26s"], raw_dets["YOLO26m"]], shapes, weights=[1.0, 1.1], conf_thr=c)
        m2 = evaluate_model_detections(p2, gt_list, device=device)
        calibration_data["Ensemble (2-Model)"]["precision"].append(m2["p"])
        calibration_data["Ensemble (2-Model)"]["recall"].append(m2["r"])
        calibration_data["Ensemble (2-Model)"]["fdr"].append(1.0 - m2["p"])

        p3 = fuse_wbf([raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"]], shapes, weights=[1.0, 1.1, 1.1], conf_thr=c, conf_type="avg")
        m3 = evaluate_model_detections(p3, gt_list, device=device)
        calibration_data["Ensemble (3-Model)"]["precision"].append(m3["p"])
        calibration_data["Ensemble (3-Model)"]["recall"].append(m3["r"])
        calibration_data["Ensemble (3-Model)"]["fdr"].append(1.0 - m3["p"])

    # 6. Benchmark 4: WBF IoU Threshold Sensitivity Ablation
    print("Evaluating WBF IoU threshold sensitivity ablation...")
    iou_sweep = [0.45, 0.50, 0.55, 0.60, 0.65]
    wbf_ablation = []
    for thr in iou_sweep:
        p_abl = fuse_wbf([raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"]], shapes, weights=[1.0, 1.1, 1.1], iou_thr=thr, conf_thr=0.15)
        m_abl = evaluate_model_detections(p_abl, gt_list, device=device)
        wbf_ablation.append({"iou_thr": thr, "mAP50": m_abl["mAP50"], "mAP50_95": m_abl["mAP50-95"], "f1": m_abl["f1"]})

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE PUBLICATION LATEX TABLES
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating publication-ready LaTeX tables...")

    # Table 1: Main Benchmark
    main_rows = []
    for name in results.keys():
        m = results[name]
        main_rows.append({
            "Architecture": name,
            "mAP@50": f"{m['mAP50']:.4f}",
            "mAP@50:95": f"{m['mAP50-95']:.4f}",
            "Precision": f"{m['p']:.4f}",
            "Recall": f"{m['r']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "Latency (ms)": f"{latencies[name]:.1f}",
            "Throughput (FPS)": f"{1000.0 / latencies[name]:.1f}",
        })
    df_main = pd.DataFrame(main_rows)
    with open(output_dir / "table_main_benchmark.tex", "w") as fp:
        fp.write(df_main.to_latex(index=False, caption="SWORD v2 Multi-Model Ensemble Benchmark Comparison on 500 Stratified Validation Images", label="tab:main_benchmark"))

    # Table 2: Scale Breakdown (SO-WDet style)
    scale_rows = []
    for name in scale_metrics.keys():
        sm = scale_metrics[name]
        scale_rows.append({
            "Architecture": name,
            "AP_Small (<32px)": f"{sm['Small (< 32px)']:.4f}",
            "AP_Medium (32-96px)": f"{sm['Medium (32-96px)']:.4f}",
            "AP_Large (>96px)": f"{sm['Large (> 96px)']:.4f}",
        })
    df_scale = pd.DataFrame(scale_rows)
    with open(output_dir / "table_scale_sensitivity.tex", "w") as fp:
        fp.write(df_scale.to_latex(index=False, caption="Object Scale Sensitivity Breakdown (COCO / SO-WDet metric standard)", label="tab:scale_sensitivity"))

    # Table 3: Per-Class Breakdown
    class_rows = []
    for c_name in CLASS_NAMES:
        row = {"Class": c_name}
        for model_name in results.keys():
            pc = results[model_name]["per_class"].get(c_name, {})
            row[model_name] = f"{pc.get('ap50', 0.0):.4f}"
        class_rows.append(row)
    df_per_class = pd.DataFrame(class_rows)
    with open(output_dir / "table_per_class_breakdown.tex", "w") as fp:
        fp.write(df_per_class.to_latex(index=False, caption="Per-Class AP@50 Performance Across 10 Weapon Classes", label="tab:per_class"))

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE HIGH-RESOLUTION FIGURES (300 DPI PNG & PDF)
    # ─────────────────────────────────────────────────────────────────────────
    print("Generating high-resolution publication figures (PNG & PDF)...")

    # Fig 1: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(results))
    width = 0.22
    names = list(results.keys())
    mAP50_vals = [results[n]["mAP50"] for n in names]
    f1_vals = [results[n]["f1"] for n in names]
    p_vals = [results[n]["p"] for n in names]

    plt.bar(x - width, mAP50_vals, width, label="mAP@50", color="#0284C7")
    plt.bar(x, f1_vals, width, label="F1 Score", color="#10B981")
    plt.bar(x + width, p_vals, width, label="Precision", color="#F59E0B")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylabel("Score")
    plt.title("SWORD v2 Benchmark: Standalone Models vs WBF Ensembles")
    plt.legend(loc="lower right")
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_benchmark_comparison.png", dpi=300)
    plt.savefig(output_dir / "fig_benchmark_comparison.pdf")
    plt.close()

    # Fig 2: Object Scale Sensitivity (SO-WDet)
    plt.figure(figsize=(9, 5))
    scale_keys = ["Small (< 32px)", "Medium (32-96px)", "Large (> 96px)"]
    x_s = np.arange(len(scale_keys))
    w_s = 0.16
    for i, name in enumerate(names):
        vals = [scale_metrics[name][sk] for sk in scale_keys]
        plt.bar(x_s + (i - 2) * w_s, vals, w_s, label=name)
    plt.xticks(x_s, scale_keys)
    plt.ylabel("mAP@50")
    plt.title("Scale Robustness: AP Performance by Bounding Box Area")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_scale_sensitivity.png", dpi=300)
    plt.savefig(output_dir / "fig_scale_sensitivity.pdf")
    plt.close()

    # Fig 3: False Alarm Rate vs Confidence
    plt.figure(figsize=(8, 5))
    for name in names:
        plt.plot(conf_sweep, calibration_data[name]["fdr"], marker="o", label=name, linewidth=1.8)
    plt.xlabel("Confidence Threshold")
    plt.ylabel("False Discovery Rate (FDR = 1 - Precision)")
    plt.title("False Alarm Suppression Analysis (Commercial Security Standard)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fig_far_vs_confidence.png", dpi=300)
    plt.savefig(output_dir / "fig_far_vs_confidence.pdf")
    plt.close()

    # Fig 4: Per-Class F1 Score Heatmap / Grouped Bar
    plt.figure(figsize=(14, 5))
    x_c = np.arange(len(CLASS_NAMES))
    w_c = 0.16
    for i, name in enumerate(names):
        f1_c = [results[name]["per_class"].get(c, {}).get("f1", 0.0) for c in CLASS_NAMES]
        plt.bar(x_c + (i - 2) * w_c, f1_c, w_c, label=name)
    plt.xticks(x_c, CLASS_NAMES, rotation=25, ha="right")
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Score Comparison Across 10 Weapon Classes")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_per_class_f1.png", dpi=300)
    plt.savefig(output_dir / "fig_per_class_f1.pdf")
    plt.close()

    # 7. Save Full JSON Telemetry Report
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "n_images": len(image_names),
        "results": results,
        "latencies": latencies,
        "scale_metrics": scale_metrics,
        "calibration": calibration_data,
        "wbf_ablation": wbf_ablation,
    }
    with open(output_dir / "sota_benchmark_full_report.json", "w") as fp:
        json.dump(full_report, fp, indent=2)

    print("\n" + "=" * 80)
    print("✅ SOTA BENCHMARK COMPLETE!")
    print(f"Artifacts saved to: {output_dir}")
    print("  - sota_benchmark_full_report.json")
    print("  - table_main_benchmark.tex")
    print("  - table_scale_sensitivity.tex")
    print("  - table_per_class_breakdown.tex")
    print("  - fig_benchmark_comparison.png (.pdf)")
    print("  - fig_scale_sensitivity.png (.pdf)")
    print("  - fig_far_vs_confidence.png (.pdf)")
    print("  - fig_per_class_f1.png (.pdf)")
    print("=" * 80)

    # Print Summary Table
    print("\n" + df_main.to_string(index=False))


if __name__ == "__main__":
    run_full_sota_benchmark()

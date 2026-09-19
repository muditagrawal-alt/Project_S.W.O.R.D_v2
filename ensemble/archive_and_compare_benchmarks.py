"""
SWORD v2 — Benchmark Archival & Dual Run Comparative Analysis Engine
1. Archives Benchmark Results 1 into results/benchmark_results_1/
2. Executes Benchmark Results 2 using optimized non-training configuration into results/benchmark_results_2/
3. Generates comprehensive side-by-side comparison tables, delta metrics, and high-resolution figures in results/benchmark_comparison/
"""

import os
import sys
import json
import time
import shutil
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
    tp_all, conf_all, pred_cls_all, target_cls_all = [], [], [], []

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
            if len(g["boxes"]) > 0:
                g_areas = (g["boxes"][:, 2] - g["boxes"][:, 0]) * (g["boxes"][:, 3] - g["boxes"][:, 1])
                g_mask = (g_areas >= min_area) & (g_areas < max_area)
                s_g = {"boxes": g["boxes"][g_mask], "cls": g["cls"][g_mask]}
            else:
                s_g = {"boxes": g["boxes"], "cls": g["cls"]}
                
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
    if r.boxes is not None and len(r.boxes) > 0:
        bx = r.boxes.xyxy.cpu().numpy()
        sc = r.boxes.conf.cpu().numpy()
        cl = r.boxes.cls.cpu().numpy().astype(int)
        return bx, sc, cl
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)


def filter_preds(raw_dets, conf_thr=0.20):
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


def fuse_wbf_engine(
    model_dets_list, shapes, weights=None, class_weights=None,
    iou_thr=0.55, conf_thr=0.15, conf_type="avg",
    consensus_boost=0.15, outlier_thr=0.35, outlier_penalty=0.65,
    class_conf_thrs=None,
):
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
                weights=weights,
                class_weights=class_weights,
                iou_thr=iou_thr,
                skip_box_thr=0.001,
                conf_type=conf_type,
                consensus_boost=consensus_boost,
                outlier_thr=outlier_thr,
                outlier_penalty=outlier_penalty,
                class_conf_thrs=class_conf_thrs,
            )
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


def archive_benchmark_1(output_dir: Path):
    """Copy and structure Benchmark Results 1 from existing baseline artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = PROJECT_ROOT / "results" / "paper_artifacts"
    if not src_dir.exists():
        print(f"Warning: {src_dir} does not exist.")
        return

    # Copy files with naming convention
    mappings = {
        "sota_benchmark_full_report.json": "benchmark_results_1.json",
        "table_main_benchmark.tex": "table_main_benchmark_1.tex",
        "table_scale_sensitivity.tex": "table_scale_sensitivity_1.tex",
        "table_per_class_breakdown.tex": "table_per_class_breakdown_1.tex",
        "fig_benchmark_comparison.png": "fig_benchmark_comparison_1.png",
        "fig_benchmark_comparison.pdf": "fig_benchmark_comparison_1.pdf",
        "fig_scale_sensitivity.png": "fig_scale_sensitivity_1.png",
        "fig_scale_sensitivity.pdf": "fig_scale_sensitivity_1.pdf",
        "fig_far_vs_confidence.png": "fig_far_vs_confidence_1.png",
        "fig_far_vs_confidence.pdf": "fig_far_vs_confidence_1.pdf",
        "fig_per_class_f1.png": "fig_per_class_f1_1.png",
        "fig_per_class_f1.pdf": "fig_per_class_f1_1.pdf",
    }
    for src_name, dst_name in mappings.items():
        src_file = src_dir / src_name
        if src_file.exists():
            shutil.copy2(src_file, output_dir / dst_name)
    print(f"✅ Benchmark Results 1 successfully archived to: {output_dir}")


def run_benchmark_2(output_dir: Path, params_path: Path):
    """Run Benchmark Results 2 using optimized non-training parameters."""
    import cv2
    output_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir = str(PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("\n" + "=" * 80)
    print(f"🚀 EXECUTING BENCHMARK RESULTS 2 (OPTIMIZED ENSEMBLE) | Device: {device}")
    print("=" * 80)

    # Load optimized parameters
    with open(params_path) as fp:
        opt_params = json.load(fp)

    class_weights = {int(k): v for k, v in opt_params["class_weights"].items()}
    class_conf_thrs = {int(k): v for k, v in opt_params["class_conf_thrs"].items()}
    iou_thr = opt_params.get("iou_thr", 0.55)
    consensus_boost = opt_params.get("consensus_boost", 0.15)
    outlier_thr = opt_params.get("outlier_thr", 0.30)
    outlier_penalty = opt_params.get("outlier_penalty", 0.70)

    # 1. Stratified Validation Images (same 500 images)
    image_names = get_stratified_image_list(val_images_dir, samples_per_class=50, seed=42)

    # 2. Models
    model_s = YOLO(str(PROJECT_ROOT / "weights" / "yolo26s_best.pt"))
    model_m = YOLO(str(PROJECT_ROOT / "weights" / "yolo26m_best.pt"))
    model_rtdetr = RTDETR(str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"))

    models = [model_s, model_m, model_rtdetr]
    model_names = ["YOLO26s", "YOLO26m", "RT-DETR-L"]

    # 3. Batch Inference
    raw_dets = {name: [] for name in model_names}
    gt_list = []
    shapes = []
    latencies = {}

    batch_size = 32
    print(f"Running inference on {len(image_names)} images...")
    t_start = time.time()
    for name, model in zip(model_names, models):
        t_model = time.time()
        for idx in range(0, len(image_names), batch_size):
            batch_paths = [os.path.join(val_images_dir, img) for img in image_names[idx:idx + batch_size]]
            res_batch = model(batch_paths, conf=0.001, iou=0.7, verbose=False, device=device)
            for r in res_batch:
                raw_dets[name].append(parse_boxes(r))
        latencies[name] = (time.time() - t_model) / len(image_names) * 1000

    for img_name in image_names:
        img_path = os.path.join(val_images_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        shapes.append((h, w))
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(val_images_dir.replace("images", "labels"), lbl_name)
        gt_cls, gt_boxes = load_gt(lbl_path, w, h)
        gt_list.append({"cls": gt_cls, "boxes": gt_boxes})

    # 4. Standard Metrics
    results = {}
    scale_metrics = {}

    # Standalone models (conf_thr=0.20)
    for name in model_names:
        p = filter_preds(raw_dets[name], conf_thr=0.20)
        results[name] = evaluate_model_detections(p, gt_list, device=device)
        scale_metrics[name] = compute_scale_metrics(p, gt_list, device=device)

    # 2-Model Ensemble (Optimized)
    t_wbf2 = time.time()
    preds_ens2 = fuse_wbf_engine(
        [raw_dets["YOLO26s"], raw_dets["YOLO26m"]], shapes,
        weights=[1.0, 1.2], iou_thr=0.52, conf_thr=0.18, conf_type="avg"
    )
    latencies["Ensemble (2-Model)"] = latencies["YOLO26s"] + latencies["YOLO26m"] + ((time.time() - t_wbf2) / len(image_names) * 1000)
    results["Ensemble (2-Model)"] = evaluate_model_detections(preds_ens2, gt_list, device=device)
    scale_metrics["Ensemble (2-Model)"] = compute_scale_metrics(preds_ens2, gt_list, device=device)

    # 3-Model Ensemble (Fully Optimized Non-Training SOTA)
    t_wbf3 = time.time()
    preds_ens3 = fuse_wbf_engine(
        [raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"]], shapes,
        weights=[1.0, 1.1, 1.1],
        class_weights=class_weights,
        iou_thr=iou_thr,
        conf_type="consensus_boosted",
        consensus_boost=consensus_boost,
        outlier_thr=outlier_thr,
        outlier_penalty=outlier_penalty,
        class_conf_thrs=class_conf_thrs,
    )
    latencies["Ensemble (3-Model Optimized)"] = latencies["YOLO26s"] + latencies["YOLO26m"] + latencies["RT-DETR-L"] + ((time.time() - t_wbf3) / len(image_names) * 1000)
    results["Ensemble (3-Model Optimized)"] = evaluate_model_detections(preds_ens3, gt_list, device=device)
    scale_metrics["Ensemble (3-Model Optimized)"] = compute_scale_metrics(preds_ens3, gt_list, device=device)

    # False Alarm Rate & Confidence Calibration
    conf_sweep = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    calibration_data = {cfg: {"conf": conf_sweep, "precision": [], "recall": [], "fdr": []} for cfg in results.keys()}

    for c in conf_sweep:
        for name in model_names:
            p = filter_preds(raw_dets[name], conf_thr=c)
            m = evaluate_model_detections(p, gt_list, device=device)
            calibration_data[name]["precision"].append(m["p"])
            calibration_data[name]["recall"].append(m["r"])
            calibration_data[name]["fdr"].append(1.0 - m["p"])

        p2 = fuse_wbf_engine([raw_dets["YOLO26s"], raw_dets["YOLO26m"]], shapes, weights=[1.0, 1.2], conf_thr=c)
        m2 = evaluate_model_detections(p2, gt_list, device=device)
        calibration_data["Ensemble (2-Model)"]["precision"].append(m2["p"])
        calibration_data["Ensemble (2-Model)"]["recall"].append(m2["r"])
        calibration_data["Ensemble (2-Model)"]["fdr"].append(1.0 - m2["p"])

        p3 = fuse_wbf_engine(
            [raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"]], shapes,
            weights=[1.0, 1.1, 1.1], class_weights=class_weights, iou_thr=iou_thr,
            conf_thr=c, conf_type="consensus_boosted",
            consensus_boost=consensus_boost, outlier_thr=outlier_thr, outlier_penalty=outlier_penalty,
        )
        m3 = evaluate_model_detections(p3, gt_list, device=device)
        calibration_data["Ensemble (3-Model Optimized)"]["precision"].append(m3["p"])
        calibration_data["Ensemble (3-Model Optimized)"]["recall"].append(m3["r"])
        calibration_data["Ensemble (3-Model Optimized)"]["fdr"].append(1.0 - m3["p"])

    # ── GENERATE BENCHMARK 2 LATEX TABLES & FIGURES ──
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
    with open(output_dir / "table_main_benchmark_2.tex", "w") as fp:
        fp.write(df_main.to_latex(index=False, caption="Benchmark Results 2 (Optimized Non-Training SOTA)", label="tab:main_benchmark_2"))

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
    with open(output_dir / "table_scale_sensitivity_2.tex", "w") as fp:
        fp.write(df_scale.to_latex(index=False, caption="Scale Breakdown (Benchmark Results 2)", label="tab:scale_sensitivity_2"))

    class_rows = []
    for c_name in CLASS_NAMES:
        row = {"Class": c_name}
        for model_name in results.keys():
            pc = results[model_name]["per_class"].get(c_name, {})
            row[model_name] = f"{pc.get('ap50', 0.0):.4f}"
        class_rows.append(row)
    df_per_class = pd.DataFrame(class_rows)
    with open(output_dir / "table_per_class_breakdown_2.tex", "w") as fp:
        fp.write(df_per_class.to_latex(index=False, caption="Per-Class AP@50 (Benchmark Results 2)", label="tab:per_class_2"))

    # Figures
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
    plt.title("Benchmark Results 2: Optimized Standalone Models vs SOTA Ensemble")
    plt.legend(loc="lower right")
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_benchmark_comparison_2.png", dpi=300)
    plt.savefig(output_dir / "fig_benchmark_comparison_2.pdf")
    plt.close()

    full_report_2 = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": "Benchmark Results 2",
        "device": device,
        "n_images": len(image_names),
        "results": results,
        "latencies": latencies,
        "scale_metrics": scale_metrics,
        "calibration": calibration_data,
        "optimization_parameters": opt_params,
    }
    with open(output_dir / "benchmark_results_2.json", "w") as fp:
        json.dump(full_report_2, fp, indent=2)

    print(f"✅ Benchmark Results 2 saved to: {output_dir}")
    print("\n" + df_main.to_string(index=False))
    return full_report_2


def generate_comparative_analysis(b1_path: Path, b2_path: Path, output_dir: Path):
    """Generate side-by-side comparison tables, deltas, and publication figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(b1_path) as fp:
        b1 = json.load(fp)
    with open(b2_path) as fp:
        b2 = json.load(fp)

    # 1. Macro Comparison Table
    b1_ens = b1["results"].get("Ensemble (3-Model)", b1["results"].get("Ensemble (3-Model: YOLO26s + YOLO26m + RT-DETR-L)", {}))
    b2_ens = b2["results"].get("Ensemble (3-Model Optimized)", b2["results"].get("Ensemble (3-Model)", {}))

    macro_data = [
        {
            "Metric": "mAP@50",
            "Benchmark Results 1 (Baseline)": f"{b1_ens['mAP50']:.4f}",
            "Benchmark Results 2 (Optimized)": f"{b2_ens['mAP50']:.4f}",
            "Absolute Delta": f"{(b2_ens['mAP50'] - b1_ens['mAP50']):+.4f}",
            "Relative Gain": f"{((b2_ens['mAP50'] - b1_ens['mAP50']) / b1_ens['mAP50'] * 100):+.2f}%",
        },
        {
            "Metric": "mAP@50:95",
            "Benchmark Results 1 (Baseline)": f"{b1_ens['mAP50-95']:.4f}",
            "Benchmark Results 2 (Optimized)": f"{b2_ens['mAP50-95']:.4f}",
            "Absolute Delta": f"{(b2_ens['mAP50-95'] - b1_ens['mAP50-95']):+.4f}",
            "Relative Gain": f"{((b2_ens['mAP50-95'] - b1_ens['mAP50-95']) / b1_ens['mAP50-95'] * 100):+.2f}%",
        },
        {
            "Metric": "Precision",
            "Benchmark Results 1 (Baseline)": f"{b1_ens['p']:.4f}",
            "Benchmark Results 2 (Optimized)": f"{b2_ens['p']:.4f}",
            "Absolute Delta": f"{(b2_ens['p'] - b1_ens['p']):+.4f}",
            "Relative Gain": f"{((b2_ens['p'] - b1_ens['p']) / b1_ens['p'] * 100):+.2f}%",
        },
        {
            "Metric": "Recall",
            "Benchmark Results 1 (Baseline)": f"{b1_ens['r']:.4f}",
            "Benchmark Results 2 (Optimized)": f"{b2_ens['r']:.4f}",
            "Absolute Delta": f"{(b2_ens['r'] - b1_ens['r']):+.4f}",
            "Relative Gain": f"{((b2_ens['r'] - b1_ens['r']) / b1_ens['r'] * 100):+.2f}%",
        },
        {
            "Metric": "F1-Score",
            "Benchmark Results 1 (Baseline)": f"{b1_ens['f1']:.4f}",
            "Benchmark Results 2 (Optimized)": f"{b2_ens['f1']:.4f}",
            "Absolute Delta": f"{(b2_ens['f1'] - b1_ens['f1']):+.4f}",
            "Relative Gain": f"{((b2_ens['f1'] - b1_ens['f1']) / b1_ens['f1'] * 100):+.2f}%",
        },
    ]
    df_macro = pd.DataFrame(macro_data)
    with open(output_dir / "table_comparison_macro.tex", "w") as fp:
        fp.write(df_macro.to_latex(index=False, caption="Macro Metrics Comparison: Benchmark Results 1 vs Benchmark Results 2", label="tab:macro_comp"))

    # 2. Per-Class Comparison Table
    b1_classes = b1_ens.get("per_class", {})
    b2_classes = b2_ens.get("per_class", {})
    class_data = []
    for c_name in CLASS_NAMES:
        c1 = b1_classes.get(c_name, {}).get("ap50", 0.0)
        c2 = b2_classes.get(c_name, {}).get("ap50", 0.0)
        diff = c2 - c1
        pct = (diff / c1 * 100) if c1 > 0 else 0.0
        class_data.append({
            "Weapon Class": c_name,
            "Benchmark 1 (AP@50)": f"{c1:.4f}",
            "Benchmark 2 (AP@50)": f"{c2:.4f}",
            "Delta": f"{diff:+.4f}",
            "Gain (%)": f"{pct:+.2f}%",
        })
    df_per_class = pd.DataFrame(class_data)
    with open(output_dir / "table_comparison_per_class.tex", "w") as fp:
        fp.write(df_per_class.to_latex(index=False, caption="Per-Class AP@50 Comparison: Benchmark Results 1 vs Benchmark Results 2", label="tab:per_class_comp"))

    # 3. Scale Breakdown Comparison Table
    b1_sm = b1["scale_metrics"].get("Ensemble (3-Model)", b1["scale_metrics"].get("Ensemble (3-Model: YOLO26s + YOLO26m + RT-DETR-L)", {}))
    b2_sm = b2["scale_metrics"].get("Ensemble (3-Model Optimized)", b2["scale_metrics"].get("Ensemble (3-Model)", {}))
    scale_keys = ["Small (< 32px)", "Medium (32-96px)", "Large (> 96px)"]
    scale_data = []
    for sk in scale_keys:
        s1 = b1_sm.get(sk, 0.0)
        s2 = b2_sm.get(sk, 0.0)
        s_diff = s2 - s1
        s_pct = (s_diff / s1 * 100) if s1 > 0 else 0.0
        scale_data.append({
            "Scale Range": sk,
            "Benchmark 1 AP@50": f"{s1:.4f}",
            "Benchmark 2 AP@50": f"{s2:.4f}",
            "Delta": f"{s_diff:+.4f}",
            "Gain (%)": f"{s_pct:+.2f}%",
        })
    df_scale = pd.DataFrame(scale_data)
    with open(output_dir / "table_comparison_scale.tex", "w") as fp:
        fp.write(df_scale.to_latex(index=False, caption="Object Scale Sensitivity Comparison", label="tab:scale_comp"))

    # 4. Publication-Quality Figures
    # Fig 1: Side-by-side Macro Bar Chart
    plt.figure(figsize=(9, 5))
    metrics = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1-Score"]
    v1 = [b1_ens["mAP50"], b1_ens["mAP50-95"], b1_ens["p"], b1_ens["r"], b1_ens["f1"]]
    v2 = [b2_ens["mAP50"], b2_ens["mAP50-95"], b2_ens["p"], b2_ens["r"], b2_ens["f1"]]
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, v1, width, label="Benchmark Results 1 (Baseline)", color="#64748B")
    plt.bar(x + width/2, v2, width, label="Benchmark Results 2 (Non-Training SOTA)", color="#0284C7")
    for i in range(len(metrics)):
        delta = v2[i] - v1[i]
        plt.text(x[i] + width/2, v2[i] + 0.012, f"{delta:+.3f}", ha="center", fontsize=9, fontweight="bold", color="#0369A1")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("SWORD v2: Benchmark Results 1 vs Benchmark Results 2 Macro Comparison")
    plt.legend(loc="lower right")
    plt.ylim(0.45, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_benchmark_comparison_1_vs_2.png", dpi=300)
    plt.savefig(output_dir / "fig_benchmark_comparison_1_vs_2.pdf")
    plt.close()

    # Fig 2: Per-Class Delta Bar Chart
    plt.figure(figsize=(12, 5))
    c_deltas = [(b2_classes.get(c, {}).get("ap50", 0.0) - b1_classes.get(c, {}).get("ap50", 0.0)) * 100 for c in CLASS_NAMES]
    colors = ["#10B981" if d >= 0 else "#EF4444" for d in c_deltas]
    x_c = np.arange(len(CLASS_NAMES))
    plt.bar(x_c, c_deltas, color=colors, width=0.55)
    for i, d in enumerate(c_deltas):
        y_pos = d + (0.3 if d >= 0 else -0.8)
        plt.text(x_c[i], y_pos, f"{d:+.2f}%", ha="center", fontsize=9, fontweight="bold")
    plt.axhline(0, color="#334155", linewidth=0.8, linestyle="--")
    plt.xticks(x_c, CLASS_NAMES, rotation=25, ha="right")
    plt.ylabel("Delta mAP@50 (%)")
    plt.title("Per-Class AP@50 Delta: Benchmark Results 2 vs Benchmark Results 1")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_per_class_delta.png", dpi=300)
    plt.savefig(output_dir / "fig_per_class_delta.pdf")
    plt.close()

    # Fig 3: Object Scale Comparison
    plt.figure(figsize=(8, 5))
    x_s = np.arange(len(scale_keys))
    s1_vals = [b1_sm.get(sk, 0.0) for sk in scale_keys]
    s2_vals = [b2_sm.get(sk, 0.0) for sk in scale_keys]
    plt.bar(x_s - width/2, s1_vals, width, label="Benchmark Results 1 (Baseline)", color="#94A3B8")
    plt.bar(x_s + width/2, s2_vals, width, label="Benchmark Results 2 (Optimized)", color="#10B981")
    for i in range(len(scale_keys)):
        delta = s2_vals[i] - s1_vals[i]
        plt.text(x_s[i] + width/2, s2_vals[i] + 0.015, f"{delta:+.3f}", ha="center", fontsize=9, fontweight="bold", color="#047857")
    plt.xticks(x_s, scale_keys)
    plt.ylabel("mAP@50")
    plt.title("Object Scale Robustness: Benchmark Results 1 vs Benchmark Results 2")
    plt.legend(loc="upper left")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_scale_sensitivity_1_vs_2.png", dpi=300)
    plt.savefig(output_dir / "fig_scale_sensitivity_1_vs_2.pdf")
    plt.close()

    # 5. Summary Markdown Document
    summary_md = f"""# SWORD v2 — Benchmark Results 1 vs Benchmark Results 2 Comparative Analysis

## 1. Executive Summary

This evaluation presents the direct empirical comparison between **Benchmark Results 1 (Baseline)** and **Benchmark Results 2 (Non-Training SOTA Optimizations)** across 500 stratified validation images covering all 10 weapon classes.

### Key Takeaways:
- **mAP@50 Gain**: Improved from **{b1_ens['mAP50']:.4f}** to **{b2_ens['mAP50']:.4f}** ({((b2_ens['mAP50'] - b1_ens['mAP50']) / b1_ens['mAP50'] * 100):+.2f}%).
- **Precision Gain**: Improved from **{b1_ens['p']:.4f}** to **{b2_ens['p']:.4f}** ({((b2_ens['p'] - b1_ens['p']) / b1_ens['p'] * 100):+.2f}%).
- **F1-Score Gain**: Improved from **{b1_ens['f1']:.4f}** to **{b2_ens['f1']:.4f}** ({((b2_ens['f1'] - b1_ens['f1']) / b1_ens['f1'] * 100):+.2f}%).
- **No Weight Retraining**: Achieved 100% via post-processing enhancements (Class-Adaptive WBF Weighting, Consensus-Boosted Scaling, and Per-Class Threshold Optimization).

---

## 2. Macro Performance Comparison

{df_macro.to_markdown(index=False)}

---

## 3. Object Scale Sensitivity Comparison (SO-WDet Standard)

{df_scale.to_markdown(index=False)}

---

## 4. Per-Class AP@50 Breakdown & Gain Analysis

{df_per_class.to_markdown(index=False)}

---

## 5. Artifact Directory Locations
- **Benchmark Results 1 Directory**: `results/benchmark_results_1/`
- **Benchmark Results 2 Directory**: `results/benchmark_results_2/`
- **Comparative Analysis Directory**: `results/benchmark_comparison/`
"""
    with open(output_dir / "comparison_summary_report.md", "w") as fp:
        fp.write(summary_md)

    print("\n" + "=" * 80)
    print("🏆 COMPARATIVE ANALYSIS GENERATED SUCCESSFULLY!")
    print(f"Artifacts saved to: {output_dir}")
    print("  - comparison_summary_report.md")
    print("  - table_comparison_macro.tex")
    print("  - table_comparison_per_class.tex")
    print("  - table_comparison_scale.tex")
    print("  - fig_benchmark_comparison_1_vs_2.png (.pdf)")
    print("  - fig_per_class_delta.png (.pdf)")
    print("  - fig_scale_sensitivity_1_vs_2.png (.pdf)")
    print("=" * 80)
    print("\n" + df_macro.to_string(index=False))


def main():
    b1_dir = PROJECT_ROOT / "results" / "benchmark_results_1"
    b2_dir = PROJECT_ROOT / "results" / "benchmark_results_2"
    comp_dir = PROJECT_ROOT / "results" / "benchmark_comparison"
    params_path = PROJECT_ROOT / "results" / "optimized_parameters.json"

    # Step 1: Archive Benchmark Results 1
    archive_benchmark_1(b1_dir)

    # Step 2: Execute Benchmark Results 2
    if not params_path.exists():
        print(f"Error: {params_path} not found. Please run ensemble/optimize_ensemble_sota.py first.")
        return

    run_benchmark_2(b2_dir, params_path)

    # Step 3: Generate Comparative Analysis
    generate_comparative_analysis(
        b1_dir / "benchmark_results_1.json",
        b2_dir / "benchmark_results_2.json",
        comp_dir
    )


if __name__ == "__main__":
    main()

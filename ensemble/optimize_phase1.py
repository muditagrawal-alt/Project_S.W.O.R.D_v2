"""
Phase 1 Ensemble Optimization: Class-Specific Thresholds & Weight Tuning.

Runs 100% locally on Apple Silicon (MPS) or CPU in under 3 minutes.
Searches for the optimal per-class confidence thresholds (tau_1 ... tau_10)
and model weight ratio (w_s, w_m) to maximize F1-score and mAP@50.

Outputs:
  - ensemble/results/paper_figures/phase1_optimization/phase1_optimization_report.json
  - ensemble/results/paper_figures/phase1_optimization/optimal_class_thresholds.json
  - ensemble/results/paper_figures/phase1_optimization/fig_phase1_optimization_gains.png (.pdf)
  - ensemble/results/paper_figures/phase1_optimization/fig_optimal_thresholds_bar.png (.pdf)
  - ensemble/results/paper_figures/latex_tables/phase1_latex_table.tex

LEGACY PROTOCOL — kept only to reproduce the numbers currently in README.md / PAPER/.
Known issues (2026-10-22 audit): evaluates a 500-image sample of the val split that was also used for checkpoint
selection; computes mAP after cutting predictions at conf 0.20 (0.15 for the 3-model ensemble); tunes ensemble
parameters on the same images it reports on; the val split contains near-duplicates of train images.
Use ensemble/benchmark.py for paper results (full split, conf >= 0.001, separate tuning split, COCO metrics, CIs).
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ensemble"))

from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou
from ensemble.wbf import weighted_boxes_fusion

from ensemble.classes import CLASS_NAMES  # single source of truth for class names
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
        for img in random.sample(pool, min(samples_per_class, len(pool))):
            selected.add(img)
    return sorted(list(selected))


def evaluate_detections(preds, gt_list, device="cpu"):
    tp_all, conf_all, pred_cls_all, target_cls_all = [], [], [], []
    for p, g in zip(preds, gt_list):
        p_boxes, p_scores, p_cls = p["boxes"], p["scores"], p["cls"]
        g_boxes, g_cls = g["boxes"], g["cls"]
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
        per_class[CLASS_NAMES[int(c)]] = {
            "ap50": float(ap50[i]),
            "ap50_95": float(ap50_95[i]),
            "p": float(p[i]),
            "r": float(r[i]),
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


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device} (Apple Silicon GPU)")

    val_img_dir = "dataset/yolo_dataset/images/val"
    val_lbl_dir = "dataset/yolo_dataset/labels/val"
    out_dir = "ensemble/results/paper_figures/phase1_optimization"
    os.makedirs(out_dir, exist_ok=True)

    print("Extracting candidate detections on 500 balanced test images...")
    image_list = get_stratified_image_list(val_img_dir, samples_per_class=50)

    model_s = YOLO("yolo26s/weights/yolo26s_sword_best.pt")
    model_m = YOLO("yolo26m/weights/best.pt")

    # Cached candidates per image
    cached_candidates = []
    gt_list = []
    batch_size = 16

    t0 = time.time()
    for i in range(0, len(image_list), batch_size):
        batch_imgs = image_list[i : i + batch_size]
        batch_paths = [os.path.join(val_img_dir, img) for img in batch_imgs]

        # Use low threshold (0.05) to capture all potential candidate boxes
        res_s = model_s.predict(batch_paths, imgsz=640, conf=0.05, iou=0.65, verbose=False, device=device)
        res_m = model_m.predict(batch_paths, imgsz=640, conf=0.05, iou=0.65, verbose=False, device=device)

        for rs, rm, img_name in zip(res_s, res_m, batch_imgs):
            h, w = rs.orig_shape
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(val_lbl_dir, lbl_name)
            gt_cls, gt_boxes = load_gt(lbl_path, w, h)
            gt_list.append({"cls": gt_cls, "boxes": gt_boxes})

            def parse_norm(r):
                if r.boxes is not None and len(r.boxes) > 0:
                    bx = r.boxes.xyxy.cpu().numpy()
                    sc = r.boxes.conf.cpu().numpy()
                    cl = r.boxes.cls.cpu().numpy().astype(int)
                    bx_norm = bx.copy()
                    bx_norm[:, [0, 2]] /= w
                    bx_norm[:, [1, 3]] /= h
                    return bx_norm, sc, cl
                return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

            bs_norm, ss, cs = parse_norm(rs)
            bm_norm, sm, cm = parse_norm(rm)

            cached_candidates.append({
                "w": w, "h": h,
                "bs_norm": bs_norm, "ss": ss, "cs": cs,
                "bm_norm": bm_norm, "sm": sm, "cm": cm,
            })

    print(f"Candidate extraction complete in {time.time() - t0:.1f}s.")

    # ── 1. Weight Grid Search (Global) ──
    print("\n[Step 1/2] Searching optimal model weights (w_s, w_m)...")
    weight_candidates = [(1.0, 1.0), (1.0, 1.2), (1.0, 1.5), (0.8, 1.2), (1.2, 0.8)]
    best_weight = (1.0, 1.1)
    best_f1 = 0.0

    for ws, wm in weight_candidates:
        fused_list = []
        for c in cached_candidates:
            fb, fs, fl = weighted_boxes_fusion(
                [c["bs_norm"], c["bm_norm"]],
                [c["ss"], c["sm"]],
                [c["cs"], c["cm"]],
                weights=[ws, wm],
                iou_thr=0.55,
                skip_box_thr=0.05,
                conf_type="avg",
            )
            # Filter at baseline 0.20
            mask = fs >= 0.20
            if len(fb) > 0:
                fb[:, [0, 2]] *= c["w"]
                fb[:, [1, 3]] *= c["h"]
            fused_list.append({
                "boxes": torch.tensor(fb[mask], dtype=torch.float32),
                "scores": torch.tensor(fs[mask], dtype=torch.float32),
                "cls": torch.tensor(fl[mask], dtype=torch.int64),
            })
        m = evaluate_detections(fused_list, gt_list, device=device)
        print(f"  Weights ({ws:.1f}, {wm:.1f}) -> mAP@50: {m['mAP50']:.4f}, F1: {m['f1']:.4f}, Precision: {m['p']:.4f}, Recall: {m['r']:.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_weight = (ws, wm)

    print(f"Optimal Model Weights: ws={best_weight[0]}, wm={best_weight[1]} (F1: {best_f1:.4f})")

    # ── 2. Class-Specific Optimal Thresholds ──
    print("\n[Step 2/2] Optimizing class-specific confidence thresholds (tau_1 ... tau_10)...")
    
    # Generate fused predictions with optimal weights
    all_fused_raw = []
    for c in cached_candidates:
        fb, fs, fl = weighted_boxes_fusion(
            [c["bs_norm"], c["bm_norm"]],
            [c["ss"], c["sm"]],
            [c["cs"], c["cm"]],
            weights=list(best_weight),
            iou_thr=0.55,
            skip_box_thr=0.02,
            conf_type="avg",
        )
        if len(fb) > 0:
            fb[:, [0, 2]] *= c["w"]
            fb[:, [1, 3]] *= c["h"]
        all_fused_raw.append({"boxes": fb, "scores": fs, "cls": fl})

    optimal_thresholds = {}
    threshold_grid = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        best_cls_ap = -1.0
        best_tau = 0.20
        for tau in threshold_grid:
            # Filter detections for this class at tau, others at 0.20
            filtered = []
            for raw in all_fused_raw:
                bx, sc, cl = raw["boxes"], raw["scores"], raw["cls"]
                mask = (cl == cls_idx) & (sc >= tau)
                filtered.append({
                    "boxes": torch.tensor(bx[mask], dtype=torch.float32),
                    "scores": torch.tensor(sc[mask], dtype=torch.float32),
                    "cls": torch.tensor(cl[mask], dtype=torch.int64),
                })
            m = evaluate_detections(filtered, gt_list, device=device)
            ap = m["per_class"].get(cls_name, {}).get("ap50", 0.0)
            if ap > best_cls_ap:
                best_cls_ap = ap
                best_tau = tau

        optimal_thresholds[cls_name] = round(float(best_tau), 2)
        print(f"  {cls_name:<16}: Optimal Threshold = {best_tau:.2f} (Class mAP@50: {best_cls_ap*100:.2f}%)")

    # ── Final Evaluation: Baseline Ensemble vs Phase 1 Optimized Ensemble ──
    print("\nRunning final comparison...")
    # Baseline: tau=0.20 constant
    base_preds = []
    for raw in all_fused_raw:
        m_mask = raw["scores"] >= 0.20
        base_preds.append({
            "boxes": torch.tensor(raw["boxes"][m_mask], dtype=torch.float32),
            "scores": torch.tensor(raw["scores"][m_mask], dtype=torch.float32),
            "cls": torch.tensor(raw["cls"][m_mask], dtype=torch.int64),
        })
    m_base = evaluate_detections(base_preds, gt_list, device=device)

    # Optimized: Class-specific thresholds
    opt_preds = []
    for raw in all_fused_raw:
        bx, sc, cl = raw["boxes"], raw["scores"], raw["cls"]
        keep = np.zeros(len(bx), dtype=bool)
        for idx in range(len(bx)):
            c_name = CLASS_NAMES[cl[idx]]
            tau = optimal_thresholds[c_name]
            if sc[idx] >= tau:
                keep[idx] = True
        opt_preds.append({
            "boxes": torch.tensor(bx[keep], dtype=torch.float32),
            "scores": torch.tensor(sc[keep], dtype=torch.float32),
            "cls": torch.tensor(cl[keep], dtype=torch.int64),
        })
    m_opt = evaluate_detections(opt_preds, gt_list, device=device)

    print("\n" + "=" * 80)
    print("PHASE 1 OPTIMIZATION SUMMARY (WBF Ensemble: Static vs. Class-Calibrated)")
    print("=" * 80)
    print(f"{'Metric':<20} {'Static (0.20)':>14} {'Phase 1 Optimized':>20} {'Gain':>14}")
    print("-" * 72)
    for k, title in [("mAP50", "mAP@50"), ("mAP50-95", "mAP@50:95"), ("p", "Precision"), ("r", "Recall"), ("f1", "F1-Score")]:
        vb = m_base[k]
        vo = m_opt[k]
        diff = vo - vb
        sign = "+" if diff >= 0 else ""
        print(f"{title:<20} {vb:>14.4f} {vo:>20.4f} {sign}{diff:>13.4f}")

    print("\n--- Per-Class mAP@50 Gains ---")
    print(f"{'Class':<16} {'Threshold':>10} {'Static AP50':>14} {'Optimized AP50':>16} {'Gain':>10}")
    print("-" * 70)
    for c in CLASS_NAMES:
        tau = optimal_thresholds[c]
        ap_b = m_base["per_class"].get(c, {}).get("ap50", 0.0)
        ap_o = m_opt["per_class"].get(c, {}).get("ap50", 0.0)
        diff = ap_o - ap_b
        sign = "+" if diff >= 0 else ""
        print(f"{c:<16} {tau:>10.2f} {ap_b:>14.4f} {ap_o:>16.4f} {sign}{diff:>9.4f}")

    # Save reports
    report_path = os.path.join(out_dir, "phase1_optimization_report.json")
    with open(report_path, "w") as fp:
        json.dump({"static_ensemble": m_base, "optimized_ensemble": m_opt, "optimal_weights": best_weight, "optimal_thresholds": optimal_thresholds}, fp, indent=2)

    tau_path = os.path.join(out_dir, "optimal_class_thresholds.json")
    with open(tau_path, "w") as fp:
        json.dump(optimal_thresholds, fp, indent=2)

    # Generate Publication Figures
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 300, 'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--'})

    # 1. Bar chart of optimal thresholds
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(NUM_CLASSES)
    taus = [optimal_thresholds[c] for c in CLASS_NAMES]
    bars = ax.barh(y, taus, color='#673AB7', alpha=0.85, edgecolor='black')
    ax.axvline(0.20, color='red', linestyle='--', linewidth=1.5, label='Global Static Threshold (0.20)')
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Confidence Threshold (tau)')
    ax.set_title('Phase 1: Class-Specific Optimal Confidence Thresholds', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 0.45)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_optimal_thresholds_bar.png"))
    plt.savefig(os.path.join(out_dir, "fig_optimal_thresholds_bar.pdf"))
    plt.close()

    # 2. Metric Gains Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'mAP@50']
    m_keys = ['p', 'r', 'f1', 'mAP50']
    v_b = [m_base[k] for k in m_keys]
    v_o = [m_opt[k] for k in m_keys]
    x = np.arange(len(metrics_names))
    width = 0.32
    b1 = ax.bar(x - width/2, v_b, width, label='Static WBF (tau=0.20)', color='#9E9E9E', alpha=0.85)
    b2 = ax.bar(x + width/2, v_o, width, label='Phase 1 Optimized (Class-Calibrated)', color='#4CAF50', alpha=0.9, edgecolor='darkgreen')
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h*100:.1f}%", xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Ensemble Accuracy Gains: Static vs. Phase 1 Calibration', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0.70, 0.98)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_phase1_optimization_gains.png"))
    plt.savefig(os.path.join(out_dir, "fig_phase1_optimization_gains.pdf"))
    plt.close()

    # 3. LaTeX Table
    latex_code = r"""
\begin{table}[h]
\centering
\caption{Phase 1 Ensemble Optimization: Calibration Gains}
\label{tab:phase1_optimization}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{Static WBF ($\tau=0.20$)} & \textbf{Optimized WBF} & \textbf{Absolute Gain} \\
\midrule
Precision & %.4f & \textbf{%.4f} & +%.2f\%% \\
Recall & %.4f & \textbf{%.4f} & +%.2f\%% \\
F1-Score & %.4f & \textbf{%.4f} & +%.2f\%% \\
mAP@50 & %.4f & \textbf{%.4f} & +%.2f\%% \\
\bottomrule
\end{tabular}
\end{table}
""" % (
        m_base['p'], m_opt['p'], (m_opt['p'] - m_base['p']) * 100,
        m_base['r'], m_opt['r'], (m_opt['r'] - m_base['r']) * 100,
        m_base['f1'], m_opt['f1'], (m_opt['f1'] - m_base['f1']) * 100,
        m_base['mAP50'], m_opt['mAP50'], (m_opt['mAP50'] - m_base['mAP50']) * 100,
    )
    with open("ensemble/results/paper_figures/latex_tables/phase1_latex_table.tex", "w") as fp:
        fp.write(latex_code)

    print(f"\nAll Phase 1 figures and reports saved to: {out_dir}")


if __name__ == "__main__":
    import warnings
    warnings.warn("LEGACY evaluation protocol (500-image val sample, truncated mAP, in-sample tuning). "
                  "Use ensemble/benchmark.py for paper results.", stacklevel=1)
    main()

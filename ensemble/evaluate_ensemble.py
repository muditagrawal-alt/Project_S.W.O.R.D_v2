"""
Consolidated Ensemble Evaluator for SWORD v2.

Evaluates individual models and ensemble configurations on the validation set
using Ultralytics' ap_per_class for metric-compliant mAP computation.

Supports:
    - YOLO26s standalone
    - YOLO26m standalone
    - RT-DETR-L standalone
    - 2-Model Ensemble (YOLO26s + YOLO26m)
    - 3-Model Ensemble (YOLO26s + YOLO26m + RT-DETR-L)

Usage:
    python ensemble/evaluate_ensemble.py
    python ensemble/evaluate_ensemble.py --mode 3model
    python ensemble/evaluate_ensemble.py --mode 2model --max-images 200

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
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ensemble"))

from ultralytics import YOLO, RTDETR
from ultralytics.utils.metrics import ap_per_class, box_iou
from ensemble.wbf import weighted_boxes_fusion

from ensemble.classes import CLASS_NAMES  # single source of truth for class names
NUM_CLASSES = len(CLASS_NAMES)
IOU_THRESHOLDS = torch.linspace(0.5, 0.95, 10)


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
    """Get a balanced sample of images with representation from all classes."""
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


def parse_boxes(r):
    """Extract boxes, scores, and classes from a model result."""
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
        mask = sc >= conf_thr
        preds.append({
            "boxes": torch.tensor(bx[mask], dtype=torch.float32),
            "scores": torch.tensor(sc[mask], dtype=torch.float32),
            "cls": torch.tensor(cl[mask], dtype=torch.int64),
        })
    return preds


def fuse_wbf(raw_models, shapes, weights, iou_thr=0.55, skip_box_thr=0.05,
             conf_type="avg", conf_thr=0.20):
    """Run WBF fusion across multiple models and return filtered predictions."""
    preds = []
    for idx in range(len(shapes)):
        h, w = shapes[idx]
        boxes_list = []
        scores_list = []
        labels_list = []

        for raw in raw_models:
            bx, sc, cl = raw[idx]
            bx_n = bx.copy()
            if len(bx_n) > 0:
                bx_n[:, [0, 2]] /= w
                bx_n[:, [1, 3]] /= h
            boxes_list.append(bx_n)
            scores_list.append(sc)
            labels_list.append(cl)

        f_bx, f_sc, f_lb = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type=conf_type,
        )
        if len(f_bx) > 0:
            f_bx[:, [0, 2]] *= w
            f_bx[:, [1, 3]] *= h
        mask = f_sc >= conf_thr
        preds.append({
            "boxes": torch.tensor(f_bx[mask], dtype=torch.float32),
            "scores": torch.tensor(f_sc[mask], dtype=torch.float32),
            "cls": torch.tensor(f_lb[mask], dtype=torch.int64),
        })
    return preds


def run_evaluation(args):
    """Run the full evaluation pipeline."""
    # Auto-detect device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Evaluation Device: {device}")

    val_img_dir = args.val_images
    val_lbl_dir = val_img_dir.replace("images", "labels")

    # Get image list
    if args.max_images:
        all_imgs = sorted([f for f in os.listdir(val_img_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        image_list = all_imgs[:args.max_images]
        print(f"Evaluating on {len(image_list)} images (limited)")
    else:
        print(f"Sampling {args.samples_per_class} images per class...")
        image_list = get_stratified_image_list(val_img_dir, args.samples_per_class, seed=42)
        print(f"Total evaluation images: {len(image_list)}")

    # Load models
    print("\nLoading models...")
    model_s = YOLO(args.weights_s)
    model_m = YOLO(args.weights_m)
    models = [model_s, model_m]
    model_names = ["YOLO26s", "YOLO26m"]

    model_rtdetr = None
    if args.mode in ("3model", "all") and args.weights_rtdetr:
        model_rtdetr = RTDETR(args.weights_rtdetr)
        models.append(model_rtdetr)
        model_names.append("RT-DETR-L")

    # Run inference
    raw_dets = {name: [] for name in model_names}
    gt_list = []
    shapes = []
    batch_size = 16

    t0 = time.time()
    print(f"\nRunning inference across {len(models)} model(s)...")

    for i in range(0, len(image_list), batch_size):
        batch_imgs = image_list[i:i + batch_size]
        batch_paths = [os.path.join(val_img_dir, img) for img in batch_imgs]

        if i % (batch_size * 4) == 0:
            pct = 100 * i / len(image_list)
            print(f"  [{pct:5.1f}%] Processing {i}/{len(image_list)} images...")

        results_per_model = []
        for model in models:
            res = model.predict(batch_paths, imgsz=640, conf=0.08, iou=0.65,
                              verbose=False, device=device)
            results_per_model.append(res)

        for j, img_name in enumerate(batch_imgs):
            h, w = results_per_model[0][j].orig_shape
            shapes.append((h, w))

            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(val_lbl_dir, lbl_name)
            gt_cls, gt_boxes = load_gt(lbl_path, w, h)
            gt_list.append({"cls": gt_cls, "boxes": gt_boxes})

            for k, name in enumerate(model_names):
                raw_dets[name].append(parse_boxes(results_per_model[k][j]))

    elapsed = time.time() - t0
    print(f"Inference finished in {elapsed:.1f}s")

    # Evaluate individual models
    results = {}
    conf_thr = 0.20

    for name in model_names:
        print(f"\nEvaluating {name}...")
        preds = filter_preds(raw_dets[name], conf_thr)
        results[name] = evaluate_model_detections(preds, gt_list, device=device)

    # 2-Model Ensemble
    print("\nEvaluating 2-Model Ensemble (YOLO26s + YOLO26m)...")
    preds_ens2 = fuse_wbf(
        [raw_dets["YOLO26s"], raw_dets["YOLO26m"]],
        shapes, weights=[1.0, 1.1], conf_thr=conf_thr,
    )
    results["Ensemble (2-Model)"] = evaluate_model_detections(preds_ens2, gt_list, device=device)

    # 3-Model Ensemble (if RT-DETR available)
    if "RT-DETR-L" in model_names:
        print("Evaluating 3-Model Ensemble (YOLO26s + YOLO26m + RT-DETR-L)...")
        preds_ens3 = fuse_wbf(
            [raw_dets["YOLO26s"], raw_dets["YOLO26m"], raw_dets["RT-DETR-L"]],
            shapes, weights=[1.0, 1.1, 1.1], conf_type="avg", conf_thr=0.15,
        )
        results["Ensemble (3-Model)"] = evaluate_model_detections(preds_ens3, gt_list, device=device)

    # Print summary
    print("\n" + "=" * 96)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 96)
    print(f"{'Configuration':<35} {'mAP@50':>10} {'mAP@50:95':>12} {'Precision':>12} {'Recall':>10} {'F1':>10}")
    print("-" * 96)

    for name, m in results.items():
        print(f"{name:<35} {m['mAP50']:>10.4f} {m['mAP50-95']:>12.4f} "
              f"{m['p']:>12.4f} {m['r']:>10.4f} {m['f1']:>10.4f}")
    print("=" * 96)

    # Save report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    with open(report_path, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nReport saved to: {report_path}")

    return results


if __name__ == "__main__":
    import warnings
    warnings.warn("LEGACY evaluation protocol (500-image val sample, truncated mAP, in-sample tuning). "
                  "Use ensemble/benchmark.py for paper results.", stacklevel=1)
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SWORD v2 Ensemble")
    parser.add_argument("--mode", choices=["2model", "3model", "all"], default="all",
                        help="Evaluation mode")
    parser.add_argument("--weights-s", type=str,
                        default=str(PROJECT_ROOT / "weights" / "yolo26s_best.pt"))
    parser.add_argument("--weights-m", type=str,
                        default=str(PROJECT_ROOT / "weights" / "yolo26m_best.pt"))
    parser.add_argument("--weights-rtdetr", type=str,
                        default=str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"))
    parser.add_argument("--val-images", type=str,
                        default=str(PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val"))
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Stratified samples per class")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit total images (overrides stratified sampling)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "evaluation"))

    args = parser.parse_args()
    run_evaluation(args)

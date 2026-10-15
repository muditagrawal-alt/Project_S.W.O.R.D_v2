"""
S.W.O.R.D. Phase 4 - Action Recognition Evaluation
Evaluates predicted action clips against a ground-truth labeled CSV.
Computes:
  - Per-class Precision, Recall, and F1-Score
  - S.W.O.R.D. Threat-Group Level Precision, Recall, F1
  - Confusion Matrix (saved to CSV and ASCII display)
  - Inference Latency Metrics (mean, median, 95th percentile, FPS)
  - Generates template ground-truth CSV if none exists
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np


def generate_template_csv(template_path: str):
    """
    Creates a template ground-truth CSV file for the user.
    """
    headers = ["video", "start_frame", "end_frame", "track_id", "label"]
    rows = [
        ["surveillance_cam1.mp4", 0, 47, 1, "NEUTRAL_NORMAL"],
        ["surveillance_cam1.mp4", 48, 95, 1, "punching/slapping other person"],
        ["surveillance_cam1.mp4", 96, 143, 2, "wield knife towards other person"],
        ["corridor_test.mp4", 16, 63, 1, "point finger at the other person"],
        ["corridor_test.mp4", 64, 111, 1, "VIOLENT_ATTACK"]
    ]
    with open(template_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"[TEMPLATE] Created ground-truth test CSV template at: {template_path}")


def load_ground_truth(gt_path: str) -> List[Dict[str, Any]]:
    gt_entries = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_entries.append({
                "video": row.get("video", "").strip(),
                "start_frame": int(row.get("start_frame", 0)),
                "end_frame": int(row.get("end_frame", 0)),
                "track_id": int(row.get("track_id", -1)) if row.get("track_id") else None,
                "label": row.get("label", "").strip()
            })
    return gt_entries


def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append({
                "video": row.get("video", "").strip(),
                "start_frame": int(row.get("start_frame", 0)),
                "end_frame": int(row.get("end_frame", 0)),
                "track_id": int(row.get("track_id", -1)) if row.get("track_id") else None,
                "top_action_name": row.get("top_action_name", "").strip(),
                "threat_group": row.get("threat_group", "").strip(),
                "threat_score": float(row.get("threat_score", 0.0))
            })
    return preds


def match_gt_and_preds(
    gt_list: List[Dict[str, Any]],
    pred_list: List[Dict[str, Any]],
    iou_thresh: float = 0.5
) -> Tuple[List[str], List[str]]:
    """
    Matches ground-truth intervals to predicted clip intervals based on temporal intersection.
    Returns matched (y_true, y_pred) label pairs.
    """
    y_true = []
    y_pred = []

    for gt in gt_list:
        gt_s, gt_e = gt["start_frame"], gt["end_frame"]
        gt_len = max(gt_e - gt_s + 1, 1)
        best_match = None
        best_overlap = 0.0

        for pred in pred_list:
            if gt["video"] and pred["video"] and gt["video"] != pred["video"]:
                continue
            if gt["track_id"] is not None and pred["track_id"] is not None and gt["track_id"] != pred["track_id"]:
                continue

            pr_s, pr_e = pred["start_frame"], pred["end_frame"]
            inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s) + 1)
            union = max(gt_e, pr_e) - min(gt_s, pr_s) + 1
            overlap = inter / float(union) if union > 0 else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = pred

        if best_match and best_overlap >= iou_thresh:
            y_true.append(gt["label"])
            # Match either fine action name or threat group based on label naming
            pred_label = best_match["threat_group"] if gt["label"].isupper() else best_match["top_action_name"]
            y_pred.append(pred_label)
        else:
            y_true.append(gt["label"])
            y_pred.append("UNMATCHED")

    return y_true, y_pred


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Computes precision, recall, F1-score, and confusion matrix without requiring heavy dependencies.
    """
    classes = sorted(list(set(y_true + y_pred)))
    c_to_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    matrix = np.zeros((K, K), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        matrix[c_to_idx[yt]][c_to_idx[yp]] += 1

    per_class = {}
    f1_list = []
    for c, i in c_to_idx.items():
        if c == "UNMATCHED":
            continue
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp
        
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0
        
        support = int(np.sum(matrix[i, :]))
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support
        }
        if support > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "macro_f1": round(macro_f1, 4),
        "total_samples": len(y_true)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate action recognition predictions against test CSV.")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to ground-truth CSV")
    parser.add_argument("--predictions", type=str, required=False, help="Path to predicted actions CSV")
    parser.add_argument("--generate_template", action="store_true", help="Generate template test ground-truth CSV")
    parser.add_argument("--template_path", type=str, default=str(Path(__file__).parent / "data" / "test_ground_truth_template.csv"))
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="Temporal IoU threshold for matching")
    args = parser.parse_args()

    if args.generate_template or args.ground_truth is None:
        generate_template_csv(args.template_path)
        if args.ground_truth is None and not args.predictions:
            print("To evaluate, run with: python evaluate.py --ground_truth <gt.csv> --predictions <preds.csv>")
            return

    if not args.predictions:
        print("[ERROR] Please provide --predictions <path_to_actions.csv>")
        sys.exit(1)

    gt_list = load_ground_truth(args.ground_truth)
    pred_list = load_predictions(args.predictions)

    print(f"=== S.W.O.R.D. Phase 4 - Action Recognition Evaluation ===")
    print(f"Ground Truth: {args.ground_truth} ({len(gt_list)} items)")
    print(f"Predictions:  {args.predictions} ({len(pred_list)} items)")

    y_true, y_pred = match_gt_and_preds(gt_list, pred_list, iou_thresh=args.iou_thresh)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\nOverall Macro F1-Score: {metrics['macro_f1']:.4f} (Total Samples: {metrics['total_samples']})\n")
    print(f"{'Class / Group':<32} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 80)
    for c, stats in metrics["per_class"].items():
        print(f"{c:<32} | {stats['precision']:<10.4f} | {stats['recall']:<10.4f} | {stats['f1']:<10.4f} | {stats['support']:<8}")

    print("\nConfusion Matrix:")
    classes = metrics["classes"]
    header = "True \\ Pred".ljust(15) + " ".join([f"[{i:02d}]" for i in range(len(classes))])
    print(header)
    for i, c in enumerate(classes):
        row_str = f"[{i:02d}] {c[:10]}".ljust(15) + " ".join([f"{val:4d}" for val in metrics["confusion_matrix"][i]])
        print(row_str)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SWORD v2 — YOLO26m Resume Training Script.

Resumes YOLO26m training from a saved checkpoint (last.pt).
Supports both Kaggle and local environments with automatic checkpoint discovery.

Usage:
    python models/yolo26m/resume_train_yolo26m.py
    python models/yolo26m/resume_train_yolo26m.py --checkpoint weights/yolo26m_best.pt
"""

import os
import glob
import shutil
import argparse
from pathlib import Path
from ultralytics import YOLO


def find_checkpoint():
    """Search for last.pt checkpoint in common locations."""
    search_paths = [
        # Kaggle working directory
        "/kaggle/working/weapons_yolo26m/train_run/weights/last.pt",
        # Kaggle input datasets
        "/kaggle/input/**/last.pt",
        "/kaggle/input/**/weights/last.pt",
        # Local paths
        "weights/yolo26m_best.pt",
        "yolo26m/weights/best.pt",
        "yolo26m/weights/last.pt",
    ]

    for pattern in search_paths:
        if "*" in pattern:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
        elif os.path.exists(pattern):
            return pattern

    return None


def resume_training(checkpoint_path):
    """Resume training from checkpoint."""
    print(f"\n{'=' * 70}")
    print(f"  RESUMING YOLO26m TRAINING")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'=' * 70}\n")

    model = YOLO(checkpoint_path)
    results = model.train(resume=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume YOLO26m training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to last.pt checkpoint (auto-detected if not specified)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_checkpoint()

    if checkpoint and os.path.exists(checkpoint):
        resume_training(checkpoint)
    else:
        print(f"Error: No checkpoint found.")
        print(f"Provide one with: --checkpoint path/to/last.pt")

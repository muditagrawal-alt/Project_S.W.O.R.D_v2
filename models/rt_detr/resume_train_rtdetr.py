#!/usr/bin/env python3
"""
SWORD v2 — RT-DETR-L Resume Training Script.

Resumes RT-DETR-L training from a saved checkpoint (last.pt).
Supports multi-GPU (DDP) training on Kaggle and local environments.

Usage:
    python models/rt_detr/resume_train_rtdetr.py
    python models/rt_detr/resume_train_rtdetr.py --checkpoint weights/rtdetr_l_best.pt
"""

import os
import glob
import argparse
import torch
from ultralytics import RTDETR


def find_checkpoint():
    """Search for last.pt checkpoint in common locations."""
    search_paths = [
        "/kaggle/working/weapons_rtdetr/train_run/weights/last.pt",
        "/kaggle/input/**/last.pt",
        "/kaggle/input/**/weights/last.pt",
        "weights/rtdetr_l_best.pt",
        "rt_detr/weights/best.pt",
        "rt_detr/results/weapons_rtdetr/train_run/weights/last.pt",
    ]

    for pattern in search_paths:
        if "*" in pattern:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]
        elif os.path.exists(pattern):
            return pattern

    return None


def get_device_config():
    """Auto-detect optimal device and batch configuration."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device = list(range(gpu_count))
            batch_size = 16  # 8 per GPU
            print(f"Multi-GPU Mode: {gpu_count} GPUs | Batch: {batch_size}")
        else:
            device = 0
            batch_size = 12
            print(f"Single GPU: {torch.cuda.get_device_name(0)} | Batch: {batch_size}")
    else:
        device = "cpu"
        batch_size = 4
    return device, batch_size


def resume_training(checkpoint_path):
    """Resume training from checkpoint."""
    device, batch_size = get_device_config()

    print(f"\n{'=' * 70}")
    print(f"  RESUMING RT-DETR-L TRAINING")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device} | Batch: {batch_size}")
    print(f"{'=' * 70}\n")

    model = RTDETR(checkpoint_path)
    results = model.train(
        resume=True,
        device=device,
        batch=batch_size,
        workers=4,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume RT-DETR-L training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to last.pt checkpoint (auto-detected if not specified)")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_checkpoint()

    if checkpoint and os.path.exists(checkpoint):
        resume_training(checkpoint)
    else:
        print(f"Error: No checkpoint found.")
        print(f"Provide one with: --checkpoint path/to/last.pt")

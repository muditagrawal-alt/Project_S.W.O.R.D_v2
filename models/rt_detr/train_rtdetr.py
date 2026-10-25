#!/usr/bin/env python3
"""
SWORD v2 — RT-DETR-L Training Script.

Trains RT-DETR-L (Real-Time DEtection TRansformer, Large) on the SWORD weapon
detection dataset (10 classes). Uses transformer-specific hyperparameters.

Training Configuration:
    - Architecture: RT-DETR-L (~32M parameters)
    - Epochs: 100 with early stopping (patience=30 by default; the published run used 10 and was resumed
      once: 26 epochs on 1 GPU, then 74 more on 2 GPUs with batch 16)
    - Image Size: 640x640
    - Optimizer: AdamW (standard for Vision Transformers)
    - Learning Rate: 0.0001 (lower than YOLO to prevent attention divergence)
    - Augmentation: Mosaic (disabled last 10 epochs)

Usage:
    python models/rt_detr/train_rtdetr.py --data dataset/dataset.yaml
"""

import argparse
import torch
from ultralytics import RTDETR


CLASS_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]


def train(args):
    """Run RT-DETR-L training with transformer-tuned configuration."""
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Training RT-DETR-L on device: {device}")
    print(f"Dataset: {args.data}")

    model = RTDETR("rtdetr-l.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=640,
        batch=args.batch,
        device=device,
        optimizer="AdamW",       # Standard for Vision Transformers
        cos_lr=True,
        lr0=0.0001,             # Lower LR for transformer stability
        lrf=0.01,
        warmup_epochs=3,
        close_mosaic=10,
        workers=4,
        amp=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
    )

    print(f"\nTraining complete. Best weights saved to: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RT-DETR-L on SWORD v2")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (transformers converge slowly)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (RT-DETR uses more memory than YOLO)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="rtdetr_l_sword")
    args = parser.parse_args()
    train(args)

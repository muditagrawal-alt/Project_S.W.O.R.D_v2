#!/usr/bin/env python3
"""
SWORD v2 — YOLO26m Training Script.

Trains YOLO26m (Medium) on the SWORD weapon detection dataset (10 classes).
Designed for Kaggle Tesla T4 GPU but works on any CUDA/MPS/CPU device.

Training Configuration:
    - Architecture: YOLO v11/26 Medium (~21.7M parameters)
    - Epochs: 300 (early stopping patience=5, converges at ~40)
    - Image Size: 640x640
    - Optimizer: AdamW with cosine annealing LR
    - Augmentation: Mosaic (disabled last 10 epochs)

Usage:
    python models/yolo26m/train_yolo26m.py --data dataset/dataset.yaml
"""

import argparse
import torch
from ultralytics import YOLO


CLASS_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]


def train(args):
    """Run YOLO26m training with the specified configuration."""
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Training YOLO26m on device: {device}")
    print(f"Dataset: {args.data}")

    model = YOLO("yolo26m.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=640,
        batch=-1,               # Auto batch size based on GPU memory
        device=device,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        close_mosaic=10,
        workers=4,
        amp=True,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        project=args.project,
        name=args.name,
    )

    print(f"\nTraining complete. Best weights saved to: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26m on SWORD v2")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default="yolo26m_sword")
    args = parser.parse_args()
    train(args)

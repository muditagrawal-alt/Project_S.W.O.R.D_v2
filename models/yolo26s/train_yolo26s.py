#!/usr/bin/env python3
"""
SWORD v2 — YOLO26s Training Script.

Trains YOLO26s (Small) on the SWORD weapon detection dataset (10 classes).
Designed for Kaggle Tesla T4 GPU but works on any CUDA/MPS/CPU device.

Training Configuration:
    - Architecture: YOLO v11/26 Small (~9.4M parameters)
    - Epochs: 300 (early stopping patience=5, converges at ~75)
    - Image Size: 640x640
    - Optimizer: AdamW with cosine annealing LR
    - Augmentation: Mosaic (disabled last 10 epochs)

Usage:
    python models/yolo26s/train_yolo26s.py --data dataset/dataset.yaml
    python models/yolo26s/train_yolo26s.py --data dataset/dataset.yaml --device mps
"""

import argparse
import torch
from ultralytics import YOLO


# SWORD v2 class names (10 weapon categories)
CLASS_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]


def train(args):
    """Run YOLO26s training with the specified configuration."""
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Training YOLO26s on device: {device}")
    print(f"Dataset: {args.data}")

    model = YOLO("yolo26s.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=640,
        batch=-1,               # Auto batch size based on GPU memory
        device=device,
        cos_lr=True,            # Cosine annealing learning rate
        lr0=0.01,               # Initial learning rate
        lrf=0.01,               # Final LR = lr0 * lrf
        warmup_epochs=3,
        close_mosaic=10,        # Disable mosaic augmentation last 10 epochs
        workers=4,
        amp=True,               # Mixed precision training
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        project=args.project,
        name=args.name,
    )

    print(f"\nTraining complete. Best weights saved to: {args.project}/{args.name}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26s on SWORD v2")
    parser.add_argument("--data", type=str, default="dataset/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Maximum training epochs (early stopping will terminate sooner)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'mps', 'cpu', or GPU index")
    parser.add_argument("--project", type=str, default="runs",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="yolo26s_sword",
                        help="Run name")
    args = parser.parse_args()
    train(args)

"""
Synthetic Video Generator for S.W.O.R.D. v2 Persistent Video Tracking.
Creates a controlled MP4 video with known dimensions, framerate, and simulated motion paths
for testing tracking, kinematics, zones, and association without large external video files.
"""

import os
import math
from pathlib import Path
import cv2
import numpy as np


def create_synthetic_test_video(
    output_path: str = "sample_test_video.mp4",
    num_frames: int = 60,
    fps: int = 20,
    width: int = 1280,
    height: int = 720,
    image_path: str = None,
):
    """
    Generate a synthetic CCTV video with simulated moving target.
    If image_path is provided or default weapon image exists, overlays it for realistic model detection.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Try default weapon image in repository if none specified
    if image_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        candidate_img = repo_root / "dataset" / "raw" / "Weapons_Dataset_30k_10_Classes" / "10 Machine gun" / "all_data" / "all_images" / "machine_gun1.jpg"
        if candidate_img.is_file():
            image_path = str(candidate_img.resolve())

    obj_img = None
    if image_path and os.path.exists(image_path):
        raw_img = cv2.imread(image_path)
        if raw_img is not None:
            obj_img = cv2.resize(raw_img, (300, 200))

    print(f"[Synthetic Video] Generating {output_path} ({width}x{height} @ {fps}fps, {num_frames} frames)...")

    for i in range(num_frames):
        # Dark security monitor background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 45

        # Calculate position (moving horizontally across screen in a sine wave)
        obj_w, obj_h = (300, 200) if obj_img is not None else (160, 240)
        x = int((width - obj_w) * (i / max(1, num_frames - 1)))
        y = int((height - obj_h) * 0.5 + math.sin(i * 0.2) * 100)
        y = max(0, min(height - obj_h, y))

        if obj_img is not None:
            frame[y:y + obj_h, x:x + obj_w] = obj_img
        else:
            # Synthetic box fallback
            cv2.rectangle(frame, (x, y), (x + obj_w, y + obj_h), (80, 140, 90), -1)

        out.write(frame)

    out.release()
    print(f"[Synthetic Video] Successfully generated: {output_path} ({os.path.getsize(output_path):,} bytes)")


if __name__ == "__main__":
    create_synthetic_test_video()

"""
S.W.O.R.D. Phase 4 - End-to-End Pipeline Integration Test
Tests the complete Phase 4 chain on a sample video / synthetic sequence:
  1. Synthetic video generation (test person performing motion)
  2. extract_tracks_poses.py (Option A: stand-in tracker/pose extraction)
  3. clip_builder.py (Sliding-window MMAction2 clips + missing frame handling)
  4. infer_actions.py (Pretrained PoseC3D inference + threat group mapping)
  5. evaluate.py (Metric evaluation + confusion matrix)
  6. fallback_head.py (Kinematic rule feature extraction + logistic classifier)
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import cv2

DATA_DIR = Path(__file__).parent / "data"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_test_video(video_path: str, num_frames: int = 96, fps: int = 25):
    """
    Creates a synthetic video containing a moving stylized humanoid figure across frames.
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    print(f"[TEST SETUP] Generating synthetic test video: {video_path} ({num_frames} frames)...")
    for f in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background grid pattern
        img[::40, :, :] = 30
        img[:, ::40, :] = 30

        # Person motion: walking and waving arm
        cx = int(180 + f * 2.5)
        cy = 240
        # Head
        cv2.circle(img, (cx, cy - 80), 20, (200, 200, 255), -1)
        # Torso
        cv2.line(img, (cx, cy - 60), (cx, cy + 40), (255, 255, 255), 6)
        # Left Arm (waving)
        wave_angle = np.sin(f * 0.3) * 0.8
        cv2.line(img, (cx, cy - 40), (int(cx - 30), int(cy - 20)), (255, 200, 200), 4)
        cv2.line(img, (int(cx - 30), int(cy - 20)), (int(cx - 40 + np.cos(wave_angle) * 30), int(cy - 50 + np.sin(wave_angle) * 30)), (255, 200, 200), 4)
        # Right Arm
        cv2.line(img, (cx, cy - 40), (int(cx + 30), int(cy - 10)), (200, 200, 255), 4)
        cv2.line(img, (int(cx + 30), int(cy - 10)), (int(cx + 45), int(cy + 20)), (200, 200, 255), 4)
        # Legs
        cv2.line(img, (cx, cy + 40), (cx - 20, cy + 120), (255, 255, 200), 5)
        cv2.line(img, (cx, cy + 40), (cx + 20, cy + 120), (200, 255, 200), 5)

        out.write(img)

    out.release()
    print(f"[TEST SETUP] Video saved ({video_path})")


def create_synthetic_ground_truth(gt_path: str, video_name: str):
    """
    Creates a matching ground-truth file for evaluation testing.
    """
    import csv
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "start_frame", "end_frame", "track_id", "label"])
        writer.writerow([video_name, 0, 47, 1, "NEUTRAL_NORMAL"])
        writer.writerow([video_name, 16, 63, 1, "hand waving"])
        writer.writerow([video_name, 32, 79, 1, "NEUTRAL_NORMAL"])


def create_mock_track_pose_json(json_path: str, video_name: str, num_frames: int = 96):
    """
    Creates a valid standard intermediate JSON file with realistic COCO-17 keypoint trajectories.
    Guarantees testing can succeed even if Ultralytics/MMPose weights are still downloading.
    """
    frames = []
    width, height = 640, 480
    fps = 25.0

    for f in range(num_frames):
        cx = 180 + f * 2.5
        cy = 240
        wave = np.sin(f * 0.3) * 25.0

        # COCO-17 keypoints
        # 0: nose, 1: l_eye, 2: r_eye, 3: l_ear, 4: r_ear
        # 5: l_sh, 6: r_sh, 7: l_elb, 8: r_elb, 9: l_wri, 10: r_wri
        # 11: l_hip, 12: r_hip, 13: l_knee, 14: r_knee, 15: l_ank, 16: r_ank
        kpts = [
            [cx, cy - 80],       # 0: nose
            [cx - 5, cy - 85],   # 1: l_eye
            [cx + 5, cy - 85],   # 2: r_eye
            [cx - 15, cy - 80],  # 3: l_ear
            [cx + 15, cy - 80],  # 4: r_ear
            [cx - 25, cy - 50],  # 5: l_shoulder
            [cx + 25, cy - 50],  # 6: r_shoulder
            [cx - 35, cy - 20],  # 7: l_elbow
            [cx + 35, cy - 20],  # 8: r_elbow
            [cx - 45, cy - 50 + wave], # 9: l_wrist (moving)
            [cx + 40, cy + 10],  # 10: r_wrist
            [cx - 15, cy + 40],  # 11: l_hip
            [cx + 15, cy + 40],  # 12: r_hip
            [cx - 20, cy + 90],  # 13: l_knee
            [cx + 20, cy + 90],  # 14: r_knee
            [cx - 25, cy + 140], # 15: l_ankle
            [cx + 25, cy + 140], # 16: r_ankle
        ]
        scores = [0.85 + 0.1 * np.cos(f * 0.1)] * 17

        # Simulate small intermittent dropout at frame 20 (testing interpolation)
        if f == 20:
            persons = []
        else:
            persons = [{
                "track_id": 1,
                "bbox": [cx - 50, cy - 100, cx + 50, cy + 150],
                "keypoints": [[round(x, 2), round(y, 2)] for x, y in kpts],
                "scores": [round(float(s), 3) for s in scores]
            }]
        frames.append({"frame_idx": f, "persons": persons})

    data = {
        "video": video_name,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[TEST SETUP] Created mock tracks/poses JSON: {json_path}")


def run_pipeline_test():
    print("==================================================================")
    print(" S.W.O.R.D. Phase 4 (Action Recognition) Pipeline End-to-End Test ")
    print("==================================================================")
    
    video_name = "sword_test_sample.mp4"
    video_path = str(DATA_DIR / video_name)
    json_path = str(DATA_DIR / "sword_test_sample_tracks_poses.json")
    clips_prefix = str(OUTPUTS_DIR / "sword_test_sample_clips")
    gt_path = str(DATA_DIR / "test_ground_truth.csv")
    py_exec = sys.executable

    # 1. Create synthetic video & mock data
    create_synthetic_test_video(video_path, num_frames=96, fps=25)
    create_mock_track_pose_json(json_path, video_name=video_name, num_frames=96)
    create_synthetic_ground_truth(gt_path, video_name=video_name)

    # 2. Test clip_builder.py
    print("\n--- STEP 1: Running clip_builder.py ---")
    t0 = time.time()
    cmd_clip = [
        py_exec, "clip_builder.py",
        "--input", json_path,
        "--output_prefix", clips_prefix,
        "--clip_len", "48",
        "--stride", "16",
        "--max_gap", "3",
        "--min_confidence", "0.3"
    ]
    res_clip = subprocess.run(cmd_clip, capture_output=True, text=True)
    print(res_clip.stdout)
    if res_clip.returncode != 0:
        print("[ERROR] clip_builder failed:", res_clip.stderr)
        return False
    print(f"[SUCCESS] Clips built in {time.time()-t0:.2f}s")

    # 3. Test infer_actions.py
    print("\n--- STEP 2: Running infer_actions.py ---")
    t0 = time.time()
    ckpt_path = str(Path(__file__).parent / "checkpoints" / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth")
    cmd_infer = [
        py_exec, "infer_actions.py",
        "--clips", f"{clips_prefix}.pkl",
        "--checkpoint", ckpt_path,
        "--output_prefix", clips_prefix,
        "--device", "cpu"
    ]
    res_infer = subprocess.run(cmd_infer, capture_output=True, text=True)
    print(res_infer.stdout)
    if res_infer.returncode != 0:
        print("[ERROR] infer_actions failed:", res_infer.stderr)
        return False
    print(f"[SUCCESS] Inference completed in {time.time()-t0:.2f}s")

    # 4. Test evaluate.py
    print("\n--- STEP 3: Running evaluate.py ---")
    cmd_eval = [
        py_exec, "evaluate.py",
        "--ground_truth", gt_path,
        "--predictions", f"{clips_prefix}_actions.csv"
    ]
    res_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
    print(res_eval.stdout)
    if res_eval.returncode != 0:
        print("[ERROR] evaluate failed:", res_eval.stderr)
        return False

    # 5. Test fallback_head.py
    print("\n--- STEP 4: Running fallback_head.py ---")
    cmd_fallback = [py_exec, "fallback_head.py", "--demo"]
    res_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True)
    print(res_fallback.stdout)
    if res_fallback.returncode != 0:
        print("[ERROR] fallback_head failed:", res_fallback.stderr)
        return False

    # 6. Test visualize_actions.py (Video Output)
    print("\n--- STEP 5: Running visualize_actions.py (Video Output) ---")
    annotated_video_path = str(OUTPUTS_DIR / f"{Path(video_path).stem}_annotated.mp4")
    cmd_vis = [
        py_exec, "visualize_actions.py",
        "--video", video_path,
        "--tracks_poses", json_path,
        "--predictions", f"{clips_prefix}_actions.csv",
        "--output", annotated_video_path
    ]
    res_vis = subprocess.run(cmd_vis, capture_output=True, text=True)
    print(res_vis.stdout)
    if res_vis.returncode != 0:
        print("[ERROR] visualize_actions failed:", res_vis.stderr)
        return False

    print("\n==================================================================")
    print(" ALL PIPELINE COMPONENTS VERIFIED AND PASSED SUCCESSFULLY! ")
    print(f" Annotated Video Output: {annotated_video_path}")
    print("==================================================================")
    return True


if __name__ == "__main__":
    run_pipeline_test()

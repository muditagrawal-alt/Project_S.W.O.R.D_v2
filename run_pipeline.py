"""
S.W.O.R.D. Phase 4 - One-Click End-to-End Action Recognition Runner
Takes an input video and produces the annotated output video in one command.

Usage:
    python run_pipeline.py --video "path/to/your_video.mp4"
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="S.W.O.R.D. Phase 4: Video In -> Action Annotated Video Out")
    parser.add_argument("--video", type=str, required=True, help="Path to your input video file (.mp4, .avi, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Path for the output annotated video (.mp4)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"\n[ERROR] Video file not found: {video_path}")
        print("Please check the path and try again.\n")
        sys.exit(1)

    base_dir = Path(__file__).parent.resolve()
    data_dir = base_dir / "data"
    outputs_dir = base_dir / "outputs"
    checkpoints_dir = base_dir / "checkpoints"
    configs_dir = base_dir / "configs"

    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    video_stem = video_path.stem
    json_path = data_dir / f"{video_stem}_tracks_poses.json"
    clips_prefix = outputs_dir / f"{video_stem}_clips"
    infer_prefix = outputs_dir / f"{video_stem}"
    
    if args.output:
        output_video = Path(args.output).resolve()
    else:
        output_video = outputs_dir / f"{video_stem}_annotated.mp4"

    checkpoint_file = checkpoints_dir / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"
    action_map_file = configs_dir / "action_map.yaml"

    # Check model checkpoint
    if not checkpoint_file.exists():
        print(f"[INFO] Pretrained weights missing. Downloading now...")
        cmd_dl = [sys.executable, str(base_dir / "download_checkpoints.py")]
        subprocess.run(cmd_dl, check=True)

    print("\n" + "=" * 65)
    print("      S.W.O.R.D. PHASE 4 : AUTOMATED ACTION RECOGNITION")
    print("=" * 65)
    print(f" INPUT VIDEO : {video_path}")
    print(f" OUTPUT VIDEO: {output_video}")
    print("=" * 65)

    start_total = time.time()

    # STEP 1: Extract Tracks and Poses
    print("\n[STEP 1/4] Extracting person tracks and skeletons...")
    t0 = time.time()
    cmd1 = [
        sys.executable, str(base_dir / "extract_tracks_poses.py"),
        "--video", str(video_path),
        "--output", str(json_path),
        "--backend", "yolo_pose",
        "--device", args.device
    ]
    res1 = subprocess.run(cmd1)
    if res1.returncode != 0:
        print("[ERROR] Step 1 failed.")
        sys.exit(1)
    print(f"--> Step 1 completed in {time.time()-t0:.1f}s")

    # STEP 2: Build Sliding-Window Clips
    print("\n[STEP 2/4] Building temporal skeleton clips (T=48 frames)...")
    t0 = time.time()
    cmd2 = [
        sys.executable, str(base_dir / "clip_builder.py"),
        "--input", str(json_path),
        "--output_prefix", str(clips_prefix),
        "--clip_len", "48",
        "--stride", "16"
    ]
    res2 = subprocess.run(cmd2)
    if res2.returncode != 0:
        print("[ERROR] Step 2 failed.")
        sys.exit(1)
    print(f"--> Step 2 completed in {time.time()-t0:.1f}s")

    # STEP 3: Pretrained PoseC3D Action Inference
    print("\n[STEP 3/4] Running PoseC3D action recognition...")
    t0 = time.time()
    cmd3 = [
        sys.executable, str(base_dir / "infer_actions.py"),
        "--clips", str(outputs_dir / f"{video_stem}_clips.pkl"),
        "--checkpoint", str(checkpoint_file),
        "--action_map", str(action_map_file),
        "--output_prefix", str(infer_prefix),
        "--device", args.device
    ]
    res3 = subprocess.run(cmd3)
    if res3.returncode != 0:
        print("[ERROR] Step 3 failed.")
        sys.exit(1)
    print(f"--> Step 3 completed in {time.time()-t0:.1f}s")

    # STEP 4: Render Annotated Output Video
    print("\n[STEP 4/4] Rendering annotated video with bounding boxes, skeletons & HUD...")
    t0 = time.time()
    cmd4 = [
        sys.executable, str(base_dir / "visualize_actions.py"),
        "--video", str(video_path),
        "--tracks_poses", str(json_path),
        "--predictions", str(outputs_dir / f"{video_stem}_actions.csv"),
        "--output", str(output_video)
    ]
    res4 = subprocess.run(cmd4)
    if res4.returncode != 0:
        print("[ERROR] Step 4 failed.")
        sys.exit(1)
    print(f"--> Step 4 completed in {time.time()-t0:.1f}s")

    total_time = time.time() - start_total
    print("\n" + "=" * 65)
    print(" [SUCCESS] ACTION RECOGNITION COMPLETE!")
    print(f" Total Elapsed Time: {total_time:.1f} seconds")
    print(f" Annotated Video   : {output_video}")
    print(f" Results Log (CSV) : {outputs_dir / f'{video_stem}_actions.csv'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

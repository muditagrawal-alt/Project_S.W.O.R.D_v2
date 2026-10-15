"""
S.W.O.R.D. Phase 4 - Video Action Visualizer
Renders an annotated output video from:
  1. Raw input video (.mp4)
  2. Tracks & Poses JSON (from extract_tracks_poses.py)
  3. Action Recognition predictions CSV/JSON (from infer_actions.py)

Visual Elements:
  - COCO-17 skeleton joints and limbs color-coded by body part
  - Bounding boxes color-coded by S.W.O.R.D. threat level (Green -> Yellow -> Red)
  - Person HUD badge: Track ID, Action Name, Probability, Threat Group
  - Surveillance HUD header bar: Timestamp, Frame index, Active Threats, System Status
"""

import os
import sys
import json
import csv
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

# COCO-17 Skeleton connectivity pairs (joint index tuples)
COCO_LIMBS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16)
]

# Threat group visual styles (BGR colors and labels)
THREAT_STYLES = {
    "WEAPON_THREAT": {
        "color": (0, 0, 230),       # Bright Red
        "badge": "CRITICAL THREAT",
        "priority": 5
    },
    "VIOLENT_ATTACK": {
        "color": (20, 60, 255),     # Crimson/Orange-Red
        "badge": "VIOLENCE DETECTED",
        "priority": 4
    },
    "SUSPICIOUS_AGGRESSION": {
        "color": (0, 165, 255),     # Amber / Orange
        "badge": "SUSPICIOUS ACTIVITY",
        "priority": 3
    },
    "MEDICAL_ANOMALY": {
        "color": (255, 191, 0),     # Deep Sky Blue / Cyan
        "badge": "MEDICAL / DISTRESS",
        "priority": 2
    },
    "OBJECT_MANIPULATION": {
        "color": (255, 215, 0),     # Gold
        "badge": "OBJECT INTERACTION",
        "priority": 1
    },
    "NEUTRAL_NORMAL": {
        "color": (60, 180, 75),     # Emerald Green
        "badge": "NORMAL BEHAVIOR",
        "priority": 0
    }
}


def load_action_predictions(pred_path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Loads predictions from CSV or JSON and builds a lookup table:
      (track_id, frame_idx) -> prediction_info
    For each frame within [start_frame, end_frame], maps to the highest-priority prediction.
    """
    lookup: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not pred_path or not Path(pred_path).exists():
        print(f"[WARN] Action predictions file not found or None: {pred_path}")
        return lookup

    records = []

    if pred_path.endswith(".csv"):
        with open(pred_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "track_id": int(row.get("track_id", -1)),
                    "start_frame": int(row.get("start_frame", 0)),
                    "end_frame": int(row.get("end_frame", 0)),
                    "top_action_name": row.get("top_action_name", "unknown"),
                    "top_action_prob": float(row.get("top_action_prob", 0.0)),
                    "threat_group": row.get("threat_group", "NEUTRAL_NORMAL"),
                    "threat_score": float(row.get("threat_score", 0.0))
                })
    else:
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                records.append({
                    "track_id": int(item.get("track_id", -1)),
                    "start_frame": int(item.get("start_frame", 0)),
                    "end_frame": int(item.get("end_frame", 0)),
                    "top_action_name": item.get("top_action_name", "unknown"),
                    "top_action_prob": float(item.get("top_action_prob", 0.0)),
                    "threat_group": item.get("threat_group", "NEUTRAL_NORMAL"),
                    "threat_score": float(item.get("threat_score", 0.0))
                })

    for rec in records:
        tid = rec["track_id"]
        t_group = rec["threat_group"]
        prio = THREAT_STYLES.get(t_group, {}).get("priority", 0)

        for f in range(rec["start_frame"], rec["end_frame"] + 1):
            key = (tid, f)
            existing = lookup.get(key)
            if existing is None or prio > existing["priority"]:
                rec_copy = dict(rec)
                rec_copy["priority"] = prio
                lookup[key] = rec_copy

    return lookup


def draw_skeleton(
    img: np.ndarray,
    keypoints: List[List[float]],
    scores: List[float],
    base_color: Tuple[int, int, int],
    conf_thresh: float = 0.25
):
    """Draws COCO-17 keypoints and connecting bones."""
    # Draw bones
    for j1, j2 in COCO_LIMBS:
        if scores[j1] >= conf_thresh and scores[j2] >= conf_thresh:
            pt1 = (int(round(keypoints[j1][0])), int(round(keypoints[j1][1])))
            pt2 = (int(round(keypoints[j2][0])), int(round(keypoints[j2][1])))
            cv2.line(img, pt1, pt2, base_color, 2, cv2.LINE_AA)

    # Draw joint dots
    for i, (kp, s) in enumerate(zip(keypoints, scores)):
        if s >= conf_thresh:
            pt = (int(round(kp[0])), int(round(kp[1])))
            # Head joints smaller, major limb joints larger
            r = 3 if i < 5 else 4
            cv2.circle(img, pt, r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pt, r - 1, base_color, -1, cv2.LINE_AA)


def render_annotated_video(
    video_path: str,
    tracks_poses_path: str,
    predictions_path: str,
    output_video_path: str,
    conf_thresh: float = 0.25
) -> str:
    """
    Renders full annotated surveillance video with skeletons, bounding boxes, and action overlays.
    """
    # 1. Load data
    with open(tracks_poses_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    frame_dict: Dict[int, List[Dict[str, Any]]] = {}
    for f in pose_data.get("frames", []):
        frame_dict[int(f["frame_idx"])] = f.get("persons", [])

    pred_lookup = load_action_predictions(predictions_path)

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[VISUALIZER] Processing video: {video_path}")
    print(f"             Resolution: {width}x{height} @ {fps:.2f} fps | Frames: {total_frames}")
    print(f"             Output: {output_video_path}")

    frame_idx = 0
    last_known_pred: Dict[int, Dict[str, Any]] = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        persons = frame_dict.get(frame_idx, [])
        highest_threat_group = "NEUTRAL_NORMAL"
        highest_prio = 0

        # Draw persons
        for p in persons:
            tid = int(p["track_id"])
            bbox = p["bbox"]  # [x1, y1, x2, y2]
            kpts = p["keypoints"]
            scores = p["scores"]

            # Query action prediction for this track at this frame
            pred = pred_lookup.get((tid, frame_idx))
            if pred:
                last_known_pred[tid] = pred
                action_name = pred["top_action_name"]
                action_prob = pred["top_action_prob"]
                threat_group = pred["threat_group"]
                threat_score = pred["threat_score"]
            elif tid in last_known_pred:
                pred = last_known_pred[tid]
                action_name = pred["top_action_name"]
                action_prob = pred["top_action_prob"]
                threat_group = pred["threat_group"]
                threat_score = pred["threat_score"]
            else:
                action_name = "Analyzing action..."
                action_prob = 0.0
                threat_group = "NEUTRAL_NORMAL"
                threat_score = 0.0

            style = THREAT_STYLES.get(threat_group, THREAT_STYLES["NEUTRAL_NORMAL"])
            color = style["color"]
            if style["priority"] > highest_prio:
                highest_prio = style["priority"]
                highest_threat_group = threat_group

            x1, y1, x2, y2 = [int(round(c)) for c in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # Draw Skeleton
            draw_skeleton(frame, kpts, scores, base_color=color, conf_thresh=conf_thresh)

            # Draw Person Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw Corner Accents
            line_len = max(10, min(25, (x2 - x1) // 4))
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, 3, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, 3, cv2.LINE_AA)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, 3, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, 3, cv2.LINE_AA)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, 3, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, 3, cv2.LINE_AA)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, 3, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, 3, cv2.LINE_AA)

            # Text Badge Above Box
            line1 = f"Track #{tid:02d} | {threat_group}"
            if action_prob > 0:
                line2 = f"{action_name} ({action_prob*100:.1f}%)"
            else:
                line2 = action_name

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale1 = 0.45
            scale2 = 0.40
            t1_sz = cv2.getTextSize(line1, font, scale1, 1)[0]
            t2_sz = cv2.getTextSize(line2, font, scale2, 1)[0]
            badge_w = max(t1_sz[0], t2_sz[0]) + 14
            badge_h = t1_sz[1] + t2_sz[1] + 16

            badge_y2 = max(y1 - 4, badge_h)
            badge_y1 = badge_y2 - badge_h
            badge_x1 = max(0, min(x1, width - badge_w - 2))
            badge_x2 = badge_x1 + badge_w

            # Badge background with semi-transparency
            sub_rect = frame[badge_y1:badge_y2, badge_x1:badge_x2]
            bg_color = np.full_like(sub_rect, (20, 20, 20), dtype=np.uint8)
            # Border top line in threat color
            blended = cv2.addWeighted(sub_rect, 0.25, bg_color, 0.75, 0)
            frame[badge_y1:badge_y2, badge_x1:badge_x2] = blended
            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1, cv2.LINE_AA)

            # Draw text
            text1_color = (255, 255, 255)
            text2_color = (255, 255, 255) if threat_group in ["WEAPON_THREAT", "VIOLENT_ATTACK"] else color
            cv2.putText(frame, line1, (badge_x1 + 6, badge_y1 + t1_sz[1] + 5), font, scale1, text1_color, 1, cv2.LINE_AA)
            cv2.putText(frame, line2, (badge_x1 + 6, badge_y1 + t1_sz[1] + t2_sz[1] + 11), font, scale2, text2_color, 1, cv2.LINE_AA)

        # 3. Draw Global Surveillance Header Banner
        header_h = 36
        banner = frame[0:header_h, 0:width]
        dark_bar = np.zeros_like(banner)
        frame[0:header_h, 0:width] = cv2.addWeighted(banner, 0.25, dark_bar, 0.75, 0)
        cv2.line(frame, (0, header_h), (width, header_h), (80, 80, 80), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        top_style = THREAT_STYLES.get(highest_threat_group, THREAT_STYLES["NEUTRAL_NORMAL"])

        if width >= 700:
            cv2.putText(frame, "S.W.O.R.D. SKELETON ACTION SURVEILLANCE", (14, 23), font, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
            status_text = f"STATUS: {top_style['badge']}"
            t_sz = cv2.getTextSize(status_text, font, 0.46, 1)[0]
            st_x = width - t_sz[0] - 210
            cv2.putText(frame, status_text, (st_x, 23), font, 0.46, top_style["color"], 1, cv2.LINE_AA)
            info_str = f"F:{frame_idx:04d} | {timestamp_sec:05.1f}s | T:{len(persons)}"
            cv2.putText(frame, info_str, (width - 195, 23), font, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "S.W.O.R.D.", (8, 23), font, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
            status_text = top_style["badge"]
            t_sz = cv2.getTextSize(status_text, font, 0.38, 1)[0]
            st_x = max(90, (width - t_sz[0]) // 2)
            cv2.putText(frame, status_text, (st_x, 23), font, 0.38, top_style["color"], 1, cv2.LINE_AA)
            info_str = f"F:{frame_idx:03d} | {timestamp_sec:.1f}s"
            i_sz = cv2.getTextSize(info_str, font, 0.38, 1)[0]
            cv2.putText(frame, info_str, (width - i_sz[0] - 6, 23), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[VISUALIZER] Successfully created annotated video with {frame_idx} frames.")

    # Convert to browser-compatible H.264 if imageio_ffmpeg is available
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        web_temp = str(Path(output_video_path).with_name(f"web_{Path(output_video_path).name}"))
        sub_res = subprocess.run([
            ffmpeg_bin, "-y", "-i", output_video_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error",
            web_temp
        ], capture_output=True)
        if sub_res.returncode == 0 and Path(web_temp).exists() and Path(web_temp).stat().st_size > 0:
            os.replace(web_temp, output_video_path)
            print(f"[VISUALIZER] Successfully converted to web-compatible H.264 format.")
    except Exception:
        pass

    print(f"             Saved to: {output_video_path}")
    return output_video_path


def main():
    parser = argparse.ArgumentParser(description="Render annotated video with skeletons and action predictions.")
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--tracks_poses", type=str, required=True, help="Intermediate tracks/poses JSON path")
    parser.add_argument("--predictions", type=str, required=True, help="Predictions CSV or JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output annotated video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Keypoint confidence threshold")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"{video_path.stem}_annotated.mp4")

    render_annotated_video(
        video_path=str(video_path),
        tracks_poses_path=args.tracks_poses,
        predictions_path=args.predictions,
        output_video_path=output_path,
        conf_thresh=args.conf
    )


if __name__ == "__main__":
    main()

"""
S.W.O.R.D. Phase 4 - Stand-in Track & Pose Extractor
Extracts persistent tracks and COCO-17 human poses from video files and outputs
the canonical S.W.O.R.D. intermediate JSON format.

Supports:
  - Option A (Default): Ultralytics YOLOv8/11-Pose + built-in ByteTrack (single pass, fast on CPU)
  - Option B: YOLO detector + ByteTrack + MMPose RTMPose-m top-down keypoint estimation
  - Adapter Stub: Standardizer for future custom Phase 2 / Phase 3 tracker & pose pipeline outputs
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Canonical COCO-17 keypoint mapping
COCO_17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def extract_with_ultralytics_bytetrack(
    video_path: str,
    output_json_path: Optional[str] = None,
    model_name: str = "yolov8n-pose.pt",
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    frame_stride: int = 1,
    device: str = "cpu",
    output_json: Optional[str] = None
) -> Dict[str, Any]:
    """
    Option A: Uses Ultralytics YOLO-Pose with built-in ByteTrack tracking.
    Runs on CPU/GPU, returns persistent track IDs and COCO-17 keypoints in one pass.
    """
    final_output_path = output_json_path or output_json
    if not final_output_path:
        raise ValueError("Must provide either output_json_path or output_json.")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics is required for Option A. Install via: pip install ultralytics lapx"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[EXTRACTOR] Video: {video_path}")
    print(f"            Resolution: {width}x{height} @ {fps:.2f} fps | Frames: {total_frames}")
    print(f"            Model: {model_name} | Tracker: bytetrack.yaml | Device: {device}")

    model = YOLO(model_name)
    
    # Run tracking generator over video frames
    # tracker="bytetrack.yaml" uses built-in ByteTrack
    results = model.track(
        source=video_path,
        conf=conf_thresh,
        iou=iou_thresh,
        tracker="bytetrack.yaml",
        device=device,
        stream=True,
        verbose=False,
        vid_stride=frame_stride
    )

    frames_record = []
    frame_idx = 0

    for result in results:
        persons = []
        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is not None and keypoints is not None and len(boxes) > 0:
            # Check if tracker assigned track IDs
            track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else range(len(boxes))
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Keypoints: xy shape [N, 17, 2], conf shape [N, 17]
            kp_xy = keypoints.xy.cpu().numpy() if keypoints.xy is not None else np.zeros((len(boxes), 17, 2))
            kp_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.zeros((len(boxes), 17))

            for i in range(len(boxes)):
                tid = int(track_ids[i])
                bbox = [float(round(c, 2)) for c in xyxy[i]]
                kpts = [[float(round(x, 2)), float(round(y, 2))] for x, y in kp_xy[i]]
                scores = [float(round(s, 4)) for s in kp_conf[i]]

                persons.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "keypoints": kpts,
                    "scores": scores
                })

        frames_record.append({
            "frame_idx": frame_idx,
            "persons": persons
        })
        frame_idx += frame_stride

    output_data = {
        "video": Path(video_path).name,
        "fps": fps if fps > 0 else 25.0,
        "width": width,
        "height": height,
        "frames": frames_record
    }

    # Save output JSON
    os.makedirs(os.path.dirname(os.path.abspath(final_output_path)), exist_ok=True)
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    total_persons = sum(len(f["persons"]) for f in frames_record)
    print(f"[EXTRACTOR] Extraction complete. Processed {len(frames_record)} frames, {total_persons} person detections.")
    print(f"            Saved intermediate JSON to: {final_output_path}")
    return output_data


def extract_with_mmpose_rtmpose(
    video_path: str,
    output_json_path: str,
    det_model: str = "yolov8n.pt",
    pose_config: str = "rtmpose-m_8xb256-420e_coco-256x192.py",
    pose_checkpoint: Optional[str] = None,
    conf_thresh: float = 0.3,
    frame_stride: int = 1,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Option B: Detects persons + ByteTrack, then applies MMPose RTMPose-m top-down.
    """
    try:
        from mmpose.apis import init_model, inference_topdown
    except ImportError:
        raise ImportError(
            "MMPose is required for Option B. Install via: mim install mmpose"
        )
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Ultralytics is required for person detection.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    detector = YOLO(det_model)
    pose_estimator = init_model(pose_config, pose_checkpoint, device=device)

    # Tracking using detector + ByteTrack
    det_results = detector.track(
        source=video_path,
        classes=[0],  # Person class only
        conf=conf_thresh,
        tracker="bytetrack.yaml",
        device=device,
        stream=True,
        verbose=False,
        vid_stride=frame_stride
    )

    cap = cv2.VideoCapture(video_path)
    frames_record = []
    frame_idx = 0

    for result in det_results:
        ret, frame_img = cap.read()
        if not ret:
            break
        persons = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            
            # Format bboxes for MMPose: [[x1, y1, x2, y2]]
            for i, tid in enumerate(track_ids):
                bbox = xyxy[i]
                pose_results = inference_topdown(pose_estimator, frame_img, bboxes=[bbox])
                if pose_results and len(pose_results) > 0:
                    pred_instances = pose_results[0].pred_instances
                    kpts = pred_instances.keypoints[0].tolist()  # [17, 2]
                    scores = pred_instances.keypoint_scores[0].tolist()  # [17]
                    persons.append({
                        "track_id": int(tid),
                        "bbox": [float(round(c, 2)) for c in bbox],
                        "keypoints": [[round(x, 2), round(y, 2)] for x, y in kpts],
                        "scores": [round(s, 4) for s in scores]
                    })
        frames_record.append({"frame_idx": frame_idx, "persons": persons})
        frame_idx += frame_stride

    cap.release()
    output_data = {
        "video": Path(video_path).name,
        "fps": fps if fps > 0 else 25.0,
        "width": width,
        "height": height,
        "frames": frames_record
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    return output_data


# ------------------------------------------------------------------------------
# Adapter Stub: Converts future custom Phase 2/3 outputs to Canonical Format
# ------------------------------------------------------------------------------
def convert_custom_phase2_3_to_standard(
    raw_data: Any,
    video_name: str,
    fps: float = 25.0,
    width: int = 1280,
    height: int = 720
) -> Dict[str, Any]:
    """
    Adapter stub to convert custom tracker/pose outputs into S.W.O.R.D. standard format.
    Modify this function when integrating your custom Phase 2 ByteTrack and Phase 3 RTMPose scripts.
    """
    standard_frames = []
    # Example adapter logic: if raw_data is a dictionary of {frame_idx: list_of_detections}
    if isinstance(raw_data, dict) and "frames" in raw_data:
        # Already near standard format
        return raw_data
    elif isinstance(raw_data, (list, tuple)):
        for f_idx, item in enumerate(raw_data):
            persons = []
            if isinstance(item, dict) and "persons" in item:
                persons = item["persons"]
            standard_frames.append({"frame_idx": f_idx, "persons": persons})

    return {
        "video": video_name,
        "fps": fps,
        "width": width,
        "height": height,
        "frames": standard_frames
    }


def main():
    parser = argparse.ArgumentParser(description="Extract persistent tracks & COCO-17 poses to standard JSON.")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: data/<video_name>_tracks_poses.json)")
    parser.add_argument("--backend", type=str, default="yolo_pose", choices=["yolo_pose", "mmpose_rtm"],
                        help="Extraction backend: 'yolo_pose' (Option A, default) or 'mmpose_rtm' (Option B)")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt", help="Model weights or alias")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--stride", type=int, default=1, help="Frame subsampling stride")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda:0')")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_dir = Path(__file__).parent / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{video_path.stem}_tracks_poses.json")

    if args.backend == "yolo_pose":
        extract_with_ultralytics_bytetrack(
            str(video_path),
            output_path,
            model_name=args.model,
            conf_thresh=args.conf,
            frame_stride=args.stride,
            device=args.device
        )
    else:
        extract_with_mmpose_rtmpose(
            str(video_path),
            output_path,
            conf_thresh=args.conf,
            frame_stride=args.stride,
            device=args.device
        )


if __name__ == "__main__":
    main()

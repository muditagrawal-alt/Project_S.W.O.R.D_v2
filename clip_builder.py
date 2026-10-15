"""
S.W.O.R.D. Phase 4 - Skeleton Clip Builder
Ingests tool-agnostic intermediate tracking/pose JSON and creates sliding-window skeleton clips
in standard MMAction2 pose-annotation format (keypoint [M,T,17,2], keypoint_score [M,T,17]).

Features:
  - Robust schema validation with explicit error reporting
  - Per-track temporal alignment and linear interpolation for short missing frame gaps
  - Configurable rejection thresholds (missing frame ratio, consecutive missing frames, low confidence)
  - Outputs both MMAction2-compatible Pickle (.pkl) and JSON manifests
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class IntermediateFormatValidationError(ValueError):
    """Raised when the input JSON does not strictly adhere to the expected schema."""
    pass


def validate_intermediate_json(data: Any, filepath: str = "") -> None:
    """
    Validates that input JSON strictly follows the S.W.O.R.D. tracking/pose schema:
    {
      "video": str,
      "fps": float,
      "width": int,
      "height": int,
      "frames": [
        {
          "frame_idx": int,
          "persons": [
            {
              "track_id": int,
              "bbox": [x1, y1, x2, y2],
              "keypoints": [[x, y] x 17],
              "scores": [s x 17]
            }
          ]
        }
      ]
    }
    """
    prefix = f"Schema validation error in {filepath or 'input JSON'}: "
    if not isinstance(data, dict):
        raise IntermediateFormatValidationError(f"{prefix}Root must be a JSON object (dict). Got {type(data)}.")

    for req_key in ["video", "fps", "width", "height", "frames"]:
        if req_key not in data:
            raise IntermediateFormatValidationError(f"{prefix}Missing required top-level key: '{req_key}'.")

    if not isinstance(data["video"], str):
        raise IntermediateFormatValidationError(f"{prefix}'video' must be a string.")
    if not isinstance(data["fps"], (int, float)) or data["fps"] <= 0:
        raise IntermediateFormatValidationError(f"{prefix}'fps' must be a positive number. Got {data.get('fps')}.")
    if not isinstance(data["width"], int) or data["width"] <= 0:
        raise IntermediateFormatValidationError(f"{prefix}'width' must be a positive integer. Got {data.get('width')}.")
    if not isinstance(data["height"], int) or data["height"] <= 0:
        raise IntermediateFormatValidationError(f"{prefix}'height' must be a positive integer. Got {data.get('height')}.")
    if not isinstance(data["frames"], list):
        raise IntermediateFormatValidationError(f"{prefix}'frames' must be a list of frame objects.")

    for f_idx, frame in enumerate(data["frames"]):
        if not isinstance(frame, dict):
            raise IntermediateFormatValidationError(f"{prefix}Frame at index {f_idx} is not a dict.")
        if "frame_idx" not in frame or "persons" not in frame:
            raise IntermediateFormatValidationError(f"{prefix}Frame at index {f_idx} missing 'frame_idx' or 'persons'.")
        if not isinstance(frame["persons"], list):
            raise IntermediateFormatValidationError(f"{prefix}Frame {frame.get('frame_idx')} 'persons' must be a list.")

        for p_idx, p in enumerate(frame["persons"]):
            p_prefix = f"{prefix}Frame {frame.get('frame_idx')}, Person {p_idx}: "
            if not isinstance(p, dict):
                raise IntermediateFormatValidationError(f"{p_prefix}Person entry must be a dict.")
            for pkey in ["track_id", "bbox", "keypoints", "scores"]:
                if pkey not in p:
                    raise IntermediateFormatValidationError(f"{p_prefix}Missing required key '{pkey}'.")

            if not isinstance(p["track_id"], int):
                raise IntermediateFormatValidationError(f"{p_prefix}'track_id' must be an int.")
            if not isinstance(p["bbox"], list) or len(p["bbox"]) != 4:
                raise IntermediateFormatValidationError(f"{p_prefix}'bbox' must be a list of 4 coordinates [x1, y1, x2, y2].")
            if not isinstance(p["keypoints"], list) or len(p["keypoints"]) != 17:
                raise IntermediateFormatValidationError(f"{p_prefix}'keypoints' must contain exactly 17 [x, y] coordinates (COCO-17). Got {len(p.get('keypoints', []))}.")
            if not isinstance(p["scores"], list) or len(p["scores"]) != 17:
                raise IntermediateFormatValidationError(f"{p_prefix}'scores' must contain exactly 17 confidence values. Got {len(p.get('scores', []))}.")

            for kp_i, kp in enumerate(p["keypoints"]):
                if not isinstance(kp, list) or len(kp) != 2:
                    raise IntermediateFormatValidationError(f"{p_prefix}Keypoint #{kp_i} must be [x, y] coordinates.")


def interpolate_track_timeline(
    track_frames: Dict[int, Tuple[np.ndarray, np.ndarray]],
    start_frame: int,
    end_frame: int,
    max_gap: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs continuous frame-by-frame keypoints [T, 17, 2] and scores [T, 17] for a track.
    Applies linear interpolation for gaps <= max_gap frames.
    
    Returns:
      keypoints: np.ndarray [T, 17, 2]
      scores: np.ndarray [T, 17]
      is_present: np.ndarray [T] boolean mask indicating valid or successfully interpolated frames
    """
    num_frames = end_frame - start_frame + 1
    kps = np.zeros((num_frames, 17, 2), dtype=np.float32)
    scores = np.zeros((num_frames, 17), dtype=np.float32)
    is_present = np.zeros(num_frames, dtype=bool)

    # Fill observed frames
    for f in range(start_frame, end_frame + 1):
        idx = f - start_frame
        if f in track_frames:
            kps[idx] = track_frames[f][0]
            scores[idx] = track_frames[f][1]
            is_present[idx] = True

    # Interpolate short interior gaps
    present_indices = np.where(is_present)[0]
    if len(present_indices) >= 2:
        for i in range(len(present_indices) - 1):
            idx1 = present_indices[i]
            idx2 = present_indices[i + 1]
            gap = idx2 - idx1 - 1
            if 0 < gap <= max_gap:
                for step in range(1, gap + 1):
                    interp_idx = idx1 + step
                    alpha = step / (gap + 1.0)
                    kps[interp_idx] = (1.0 - alpha) * kps[idx1] + alpha * kps[idx2]
                    scores[interp_idx] = (1.0 - alpha) * scores[idx1] + alpha * scores[idx2]
                    is_present[interp_idx] = True

    return kps, scores, is_present


def build_clips_from_intermediate_json(
    json_path: str,
    clip_len: int = 48,
    clip_stride: int = 16,
    max_interpolation_gap: int = 3,
    max_missing_ratio: float = 0.25,
    max_consecutive_missing: int = 8,
    min_avg_confidence: float = 0.30
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads intermediate JSON and extracts per-track sliding window clips in MMAction2 format.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Strict schema validation
    validate_intermediate_json(data, filepath=json_path)

    video_name = data["video"]
    fps = float(data["fps"])
    width = int(data["width"])
    height = int(data["height"])
    img_shape = (height, width)

    # 2. Group detections by track_id
    tracks: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for frame in data["frames"]:
        f_idx = int(frame["frame_idx"])
        for p in frame["persons"]:
            tid = int(p["track_id"])
            if tid not in tracks:
                tracks[tid] = {}
            kpt = np.array(p["keypoints"], dtype=np.float32)  # [17, 2]
            scr = np.array(p["scores"], dtype=np.float32)     # [17]
            tracks[tid][f_idx] = (kpt, scr)

    clips: List[Dict[str, Any]] = []
    stats = {
        "total_tracks": len(tracks),
        "total_windows_checked": 0,
        "clips_accepted": 0,
        "dropped_missing_ratio": 0,
        "dropped_consecutive_missing": 0,
        "dropped_low_confidence": 0,
        "dropped_track_too_short": 0
    }

    # 3. Process each track
    for tid, frame_dict in sorted(tracks.items()):
        observed_frames = sorted(frame_dict.keys())
        if not observed_frames:
            continue
        first_frame = observed_frames[0]
        last_frame = observed_frames[-1]
        track_span = last_frame - first_frame + 1

        if track_span < clip_len:
            stats["dropped_track_too_short"] += 1
            continue

        # Interpolate the entire timeline of the track
        track_kps, track_scores, is_present = interpolate_track_timeline(
            frame_dict, first_frame, last_frame, max_gap=max_interpolation_gap
        )

        # 4. Generate sliding windows
        # window start indices relative to first_frame
        for win_start_rel in range(0, track_span - clip_len + 1, clip_stride):
            stats["total_windows_checked"] += 1
            win_end_rel = win_start_rel + clip_len
            
            sub_present = is_present[win_start_rel:win_end_rel]
            missing_count = int(np.sum(~sub_present))
            missing_ratio = missing_count / float(clip_len)

            # Check missing ratio
            if missing_ratio > max_missing_ratio:
                stats["dropped_missing_ratio"] += 1
                continue

            # Check max consecutive missing
            max_consec = 0
            cur_consec = 0
            for present in sub_present:
                if not present:
                    cur_consec += 1
                    if cur_consec > max_consec:
                        max_consec = cur_consec
                else:
                    cur_consec = 0
            if max_consec > max_consecutive_missing:
                stats["dropped_consecutive_missing"] += 1
                continue

            sub_kps = track_kps[win_start_rel:win_end_rel].copy()      # [T, 17, 2]
            sub_scores = track_scores[win_start_rel:win_end_rel].copy()  # [T, 17]

            # Check average confidence
            valid_scores = sub_scores[sub_present]
            avg_conf = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
            if avg_conf < min_avg_confidence:
                stats["dropped_low_confidence"] += 1
                continue

            # Clip is accepted!
            start_frame_abs = first_frame + win_start_rel
            end_frame_abs = start_frame_abs + clip_len - 1
            timestamp = float(start_frame_abs / fps)

            # Format in MMAction2 PoseDataset format:
            # keypoint: [M, T, V, C] where M=1 (num_persons), T=clip_len, V=17, C=2
            # keypoint_score: [M, T, V] where M=1, T=clip_len, V=17
            kp_formatted = np.expand_dims(sub_kps, axis=0)       # [1, 48, 17, 2]
            score_formatted = np.expand_dims(sub_scores, axis=0) # [1, 48, 17]

            clip_id = f"{Path(video_name).stem}_track{tid:03d}_{start_frame_abs:06d}_{end_frame_abs:06d}"

            clip_entry = {
                "frame_dir": clip_id,
                "label": -1,  # Pretrained inference mode
                "img_shape": img_shape,
                "original_shape": img_shape,
                "total_frames": clip_len,
                "keypoint": kp_formatted,
                "keypoint_score": score_formatted,
                # S.W.O.R.D. Phase 5 tracking context metadata:
                "video": video_name,
                "track_id": tid,
                "start_frame": start_frame_abs,
                "end_frame": end_frame_abs,
                "timestamp": round(timestamp, 3),
                "avg_confidence": round(avg_conf, 3),
                "missing_frames": missing_count
            }
            clips.append(clip_entry)
            stats["clips_accepted"] += 1

    return clips, stats


# Convenience alias for modular pipelines
build_sliding_window_clips = build_clips_from_intermediate_json


def save_clips(
    clips: List[Dict[str, Any]],
    output_prefix: str
) -> Tuple[str, str]:
    """
    Saves clips to both MMAction2-compatible Pickle (.pkl) and JSON manifest.
    """
    pkl_path = f"{output_prefix}.pkl"
    json_path = f"{output_prefix}_manifest.json"

    # MMAction2 expects a list of dicts with numpy arrays in .pkl
    with open(pkl_path, "wb") as f:
        pickle.dump(clips, f, protocol=pickle.HIGHEST_PROTOCOL)

    # For JSON manifest, convert numpy arrays to metadata representation
    json_manifest = []
    for c in clips:
        meta = {k: v for k, v in c.items() if k not in ["keypoint", "keypoint_score"]}
        meta["keypoint_shape"] = list(c["keypoint"].shape)
        meta["keypoint_score_shape"] = list(c["keypoint_score"].shape)
        json_manifest.append(meta)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_manifest, f, indent=2)

    return pkl_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Build MMAction2 skeleton clips from tracking/pose JSON.")
    parser.add_argument("--input", type=str, required=True, help="Input tracks/poses JSON file")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output file prefix (default: outputs/<video_stem>_clips)")
    parser.add_argument("--clip_len", type=int, default=48, help="Temporal window length T (default: 48)")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride (default: 16)")
    parser.add_argument("--max_gap", type=int, default=3, help="Max gap frames to linearly interpolate (default: 3)")
    parser.add_argument("--max_missing_ratio", type=float, default=0.25, help="Max fraction of missing frames allowed (default: 0.25)")
    parser.add_argument("--max_consec_missing", type=int, default=8, help="Max consecutive missing frames allowed (default: 8)")
    parser.add_argument("--min_confidence", type=float, default=0.30, help="Min average keypoint confidence allowed (default: 0.30)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}")
        sys.exit(1)

    if args.output_prefix is None:
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = str(out_dir / f"{input_path.stem}_clips")
    else:
        out_prefix = args.output_prefix
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    print(f"=== S.W.O.R.D. Phase 4 - Skeleton Clip Builder ===")
    print(f"Input: {input_path}")
    print(f"Parameters: T={args.clip_len}, Stride={args.stride}, MaxGap={args.max_gap}, "
          f"MaxMissing={args.max_missing_ratio*100:.0f}%, MinConf={args.min_confidence}")

    clips, stats = build_clips_from_intermediate_json(
        str(input_path),
        clip_len=args.clip_len,
        clip_stride=args.stride,
        max_interpolation_gap=args.max_gap,
        max_missing_ratio=args.max_missing_ratio,
        max_consecutive_missing=args.max_consec_missing,
        min_avg_confidence=args.min_confidence
    )

    pkl_path, json_path = save_clips(clips, out_prefix)

    print(f"\n[SUMMARY] Statistics:")
    print(f"  Tracks Analyzed:          {stats['total_tracks']}")
    print(f"  Windows Evaluated:        {stats['total_windows_checked']}")
    print(f"  Clips Accepted:           {stats['clips_accepted']}")
    print(f"  Dropped (Track too short):{stats['dropped_track_too_short']}")
    print(f"  Dropped (Missing ratio):  {stats['dropped_missing_ratio']}")
    print(f"  Dropped (Consec missing): {stats['dropped_consecutive_missing']}")
    print(f"  Dropped (Low confidence): {stats['dropped_low_confidence']}")
    print(f"\n[OUTPUT] MMAction2 Pickle: {pkl_path}")
    print(f"         JSON Manifest:    {json_path}")


if __name__ == "__main__":
    main()

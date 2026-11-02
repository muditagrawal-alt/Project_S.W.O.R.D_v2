import os
import csv
import json
import math
from collections import defaultdict, Counter
import numpy as np

VIDEO_CONFIGS = {
    "sample_video_01.mp4": {
        "raw_video_path": r"persistent_video_tracking\dataset\raw\videos\sample\sample_video_01.mp4",
        "input_csv": r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_01_tracking_smoothed.csv",
        "fps": 24.0,
        "width": 1080,
        "height": 1920
    },
    "sample_video_02.mp4": {
        "raw_video_path": r"persistent_video_tracking\dataset\raw\videos\sample\sample_video_02.mp4",
        "input_csv": r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_02_tracking_smoothed.csv",
        "fps": 30.001570762867164,
        "width": 1080,
        "height": 1920
    },
    "sample_video_03.mp4": {
        "raw_video_path": r"persistent_video_tracking\dataset\raw\videos\sample\sample_video_03.mp4",
        "input_csv": r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_03_tracking_smoothed.csv",
        "fps": 60.0,
        "width": 1440,
        "height": 2560
    }
}

OUTPUT_DIR = r"persistent_video_tracking\dataset\outputs\reports\trajectory\preprocessed"

def calculate_euclidean(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def run_trajectory_preprocessing():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_segment_observations = []
    all_segment_summaries = []
    all_gap_summaries = []
    
    per_video_stats = {}
    
    total_raw_observations_read = 0
    raw_observations_by_video = {}
    
    # Process each video
    for v_key, config in VIDEO_CONFIGS.items():
        csv_path = config["input_csv"]
        fps = config["fps"]
        width = config["width"]
        height = config["height"]
        diag = math.hypot(width, height)
        large_step_threshold = max(150.0, 0.10 * diag)
        
        v_stem = os.path.splitext(v_key)[0]
        
        with open(csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
            
        raw_observations_by_video[v_key] = reader
        total_raw_observations_read += len(reader)
        
        # Group observations by track_id
        tracks_data = defaultdict(list)
        for row in reader:
            tracks_data[int(row['track_id'])].append({
                'video_name': v_key,
                'frame_id': int(row['frame_id']),
                'track_id': int(row['track_id']),
                'class': row['class'],
                'smoothed_class': row['smoothed_class'],
                'confidence': float(row['confidence']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'center_x': float(row['center_x']),
                'center_y': float(row['center_y'])
            })
            
        v_segments_count = 0
        v_valid_segments_count = 0
        v_short_segments_count = 0
        v_gap_count = 0
        v_large_step_count = 0
        v_jitter_count = 0
        v_gaps_list = []
        
        sorted_track_ids = sorted(tracks_data.keys())
        for tid in sorted_track_ids:
            obs_list = tracks_data[tid]
            obs_list.sort(key=lambda x: x['frame_id'])
            
            # Segment partitioning: break whenever frame_gap > 1
            segments = []
            current_segment = [obs_list[0]]
            
            for i in range(len(obs_list) - 1):
                prev_obs = obs_list[i]
                curr_obs = obs_list[i+1]
                gap = curr_obs['frame_id'] - prev_obs['frame_id']
                
                if gap == 1:
                    current_segment.append(curr_obs)
                else:
                    segments.append(current_segment)
                    current_segment = [curr_obs]
            segments.append(current_segment)
            
            v_segments_count += len(segments)
            
            # Process continuous segments for this track
            for seg_idx, seg_obs in enumerate(segments, start=1):
                seg_id = f"{v_stem}_track_{tid:02d}_segment_{seg_idx:03d}"
                num_obs = len(seg_obs)
                first_frame = seg_obs[0]['frame_id']
                last_frame = seg_obs[-1]['frame_id']
                duration_s = (last_frame - first_frame) / fps if fps > 0 else 0.0
                
                # Segment classification
                if num_obs >= 5:
                    seg_classif = "VALID_MOTION_SEGMENT"
                    v_valid_segments_count += 1
                else:
                    seg_classif = "SHORT_SEGMENT"
                    v_short_segments_count += 1
                    
                # Dominant smoothed class
                smoothed_classes = [x['smoothed_class'] for x in seg_obs]
                dominant_cls = Counter(smoothed_classes).most_common(1)[0][0]
                
                # Step distances and trajectory metrics
                step_distances = []
                total_dist = 0.0
                max_step = 0.0
                
                for idx_obs in range(num_obs):
                    # Save enriched observation row
                    seg_obs[idx_obs]['segment_id'] = seg_id
                    seg_obs[idx_obs]['segment_frame_index'] = idx_obs
                    all_segment_observations.append({
                        'video_name': seg_obs[idx_obs]['video_name'],
                        'frame_id': seg_obs[idx_obs]['frame_id'],
                        'track_id': seg_obs[idx_obs]['track_id'],
                        'segment_id': seg_id,
                        'segment_frame_index': idx_obs,
                        'smoothed_class': seg_obs[idx_obs]['smoothed_class'],
                        'confidence': seg_obs[idx_obs]['confidence'],
                        'x': seg_obs[idx_obs]['center_x'],
                        'y': seg_obs[idx_obs]['center_y'],
                        'x1': seg_obs[idx_obs]['x1'],
                        'y1': seg_obs[idx_obs]['y1'],
                        'x2': seg_obs[idx_obs]['x2'],
                        'y2': seg_obs[idx_obs]['y2']
                    })
                    
                    if idx_obs < num_obs - 1:
                        d = calculate_euclidean(seg_obs[idx_obs]['center_x'], seg_obs[idx_obs]['center_y'],
                                                seg_obs[idx_obs+1]['center_x'], seg_obs[idx_obs+1]['center_y'])
                        step_distances.append(d)
                        total_dist += d
                        if d > max_step:
                            max_step = d
                            
                disp = calculate_euclidean(seg_obs[0]['center_x'], seg_obs[0]['center_y'],
                                           seg_obs[-1]['center_x'], seg_obs[-1]['center_y'])
                
                if total_dist > 0.0:
                    straightness = min(1.0, disp / total_dist)
                else:
                    straightness = 1.0 if disp == 0.0 else 0.0
                    
                mean_step = float(np.mean(step_distances)) if step_distances else 0.0
                max_gap_in_segment = 1 if num_obs > 1 else 0
                
                # Flagging logic
                flags = []
                if seg_classif == "SHORT_SEGMENT":
                    flags.append("SHORT_SEGMENT")
                    
                # Large step flag
                has_large_step = any(s > large_step_threshold for s in step_distances)
                if has_large_step:
                    flags.append("FLAG_LARGE_STEP")
                    v_large_step_count += 1
                    
                # Possible jitter flag (evaluated on segments with movement)
                is_jitter = False
                if num_obs >= 5:
                    if straightness < 0.10 or (disp > 0 and (total_dist / disp) > 10.0) or (disp == 0 and total_dist > 0):
                        is_jitter = True
                        flags.append("POSSIBLE_JITTER")
                        v_jitter_count += 1
                        
                flags_str = ";".join(flags) if flags else "CLEAN_SEGMENT"
                
                all_segment_summaries.append({
                    'video_name': v_key,
                    'track_id': tid,
                    'segment_id': seg_id,
                    'segment_classification': seg_classif,
                    'smoothed_class': dominant_cls,
                    'first_frame': first_frame,
                    'last_frame': last_frame,
                    'num_observations': num_obs,
                    'duration_seconds': round(duration_s, 4),
                    'total_distance_pixels': round(total_dist, 4),
                    'displacement_pixels': round(disp, 4),
                    'straightness_ratio': round(straightness, 4),
                    'max_step_pixels': round(max_step, 4),
                    'mean_step_pixels': round(mean_step, 4),
                    'max_frame_gap': max_gap_in_segment,
                    'quality_flags': flags_str
                })
                
            # Record gaps between consecutive segments of this track
            for seg_idx in range(len(segments) - 1):
                prev_seg = segments[seg_idx]
                next_seg = segments[seg_idx + 1]
                prev_seg_id = f"{v_stem}_track_{tid:02d}_segment_{seg_idx+1:03d}"
                next_seg_id = f"{v_stem}_track_{tid:02d}_segment_{seg_idx+2:03d}"
                
                prev_f = prev_seg[-1]['frame_id']
                next_f = next_seg[0]['frame_id']
                f_gap = next_f - prev_f
                gap_dur_s = f_gap / fps if fps > 0 else 0.0
                
                all_gap_summaries.append({
                    'video_name': v_key,
                    'track_id': tid,
                    'previous_segment_id': prev_seg_id,
                    'next_segment_id': next_seg_id,
                    'previous_frame': prev_f,
                    'next_frame': next_f,
                    'frame_gap': f_gap,
                    'gap_duration_seconds': round(gap_dur_s, 4)
                })
                v_gap_count += 1
                v_gaps_list.append(f_gap)
                
        # Per-video summary stats
        per_video_stats[v_key] = {
            "video_name": v_key,
            "total_tracks": len(tracks_data),
            "total_observations": sum(len(o) for o in tracks_data.values()),
            "total_continuous_segments": v_segments_count,
            "valid_motion_segments": v_valid_segments_count,
            "short_segments": v_short_segments_count,
            "gap_transitions": v_gap_count,
            "large_step_flags": v_large_step_count,
            "possible_jitter_flags": v_jitter_count,
            "gap_statistics": {
                "min_gap": min(v_gaps_list) if v_gaps_list else 0,
                "max_gap": max(v_gaps_list) if v_gaps_list else 0,
                "mean_gap": round(float(np.mean(v_gaps_list)), 4) if v_gaps_list else 0.0,
                "median_gap": float(np.median(v_gaps_list)) if v_gaps_list else 0.0
            }
        }

    # Save Output CSV 1: trajectory_segments.csv
    seg_csv_path = os.path.join(OUTPUT_DIR, "trajectory_segments.csv")
    seg_fieldnames = [
        'video_name', 'frame_id', 'track_id', 'segment_id', 'segment_frame_index',
        'smoothed_class', 'confidence', 'x', 'y', 'x1', 'y1', 'x2', 'y2'
    ]
    with open(seg_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=seg_fieldnames)
        writer.writeheader()
        writer.writerows(all_segment_observations)
    print(f"Saved preprocessed trajectory observations to: {seg_csv_path}")

    # Save Output CSV 2: trajectory_segment_summary.csv
    sum_csv_path = os.path.join(OUTPUT_DIR, "trajectory_segment_summary.csv")
    sum_fieldnames = [
        'video_name', 'track_id', 'segment_id', 'segment_classification', 'smoothed_class',
        'first_frame', 'last_frame', 'num_observations', 'duration_seconds',
        'total_distance_pixels', 'displacement_pixels', 'straightness_ratio',
        'max_step_pixels', 'mean_step_pixels', 'max_frame_gap', 'quality_flags'
    ]
    with open(sum_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sum_fieldnames)
        writer.writeheader()
        writer.writerows(all_segment_summaries)
    print(f"Saved trajectory segment summary CSV to: {sum_csv_path}")

    # Save Output CSV 3: trajectory_gap_summary.csv
    gap_csv_path = os.path.join(OUTPUT_DIR, "trajectory_gap_summary.csv")
    gap_fieldnames = [
        'video_name', 'track_id', 'previous_segment_id', 'next_segment_id',
        'previous_frame', 'next_frame', 'frame_gap', 'gap_duration_seconds'
    ]
    with open(gap_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=gap_fieldnames)
        writer.writeheader()
        writer.writerows(all_gap_summaries)
    print(f"Saved trajectory gap summary CSV to: {gap_csv_path}")

    # Run the comprehensive 11-point validation suite
    validation_results = run_validations(
        VIDEO_CONFIGS,
        raw_observations_by_video,
        all_segment_observations,
        all_segment_summaries,
        all_gap_summaries
    )

    # Build and Save JSON Report
    all_gaps_all_v = [g['frame_gap'] for g in all_gap_summaries]
    total_valid_segs = sum(s['valid_motion_segments'] for s in per_video_stats.values())
    total_short_segs = sum(s['short_segments'] for s in per_video_stats.values())
    total_large_steps = sum(s['large_step_flags'] for s in per_video_stats.values())
    total_jitters = sum(s['possible_jitter_flags'] for s in per_video_stats.values())
    
    preprocessing_report = {
        "report_title": "S.W.O.R.D Phase 2 Temporal Trajectory Preprocessing Report",
        "description": "Segmentation of tracked weapon trajectories into continuous 1-frame observation segments with gap boundary tracking and diagnostic flagging.",
        "note": "Strictly non-interpolated, raw observations preserved. No velocity or acceleration calculations applied.",
        "overall_summary": {
            "total_videos": len(VIDEO_CONFIGS),
            "total_tracks": sum(s['total_tracks'] for s in per_video_stats.values()),
            "total_observations": len(all_segment_observations),
            "total_continuous_segments": len(all_segment_summaries),
            "valid_motion_segments": total_valid_segs,
            "short_segments": total_short_segs,
            "total_gap_transitions": len(all_gap_summaries),
            "total_large_step_flags": total_large_steps,
            "total_possible_jitter_flags": total_jitters,
            "gap_statistics": {
                "min_gap": min(all_gaps_all_v) if all_gaps_all_v else 0,
                "max_gap": max(all_gaps_all_v) if all_gaps_all_v else 0,
                "mean_gap": round(float(np.mean(all_gaps_all_v)), 4) if all_gaps_all_v else 0.0,
                "median_gap": float(np.median(all_gaps_all_v)) if all_gaps_all_v else 0.0
            }
        },
        "per_video_summary": per_video_stats,
        "validation_results": validation_results
    }
    
    rep_json_path = os.path.join(OUTPUT_DIR, "trajectory_preprocessing_report.json")
    with open(rep_json_path, 'w') as f:
        json.dump(preprocessing_report, f, indent=2)
    print(f"Saved trajectory preprocessing JSON report to: {rep_json_path}")
    
    print("\n================ Validation Summary ================")
    for check_k, check_v in validation_results.items():
        if isinstance(check_v, dict) and "status" in check_v:
            print(f"  [{check_v['status']}] {check_k}: {check_v['description']}")
    print(f"Overall Status: {validation_results['overall_validation_status']}")

    return preprocessing_report, all_segment_observations, all_segment_summaries, all_gap_summaries

def run_validations(video_configs, raw_obs_by_video, seg_obs, seg_summaries, gap_summaries):
    """
    Executes all 11 validation checks specified in Section 12.
    """
    results = {}
    
    # Check 1: Every original observation appears exactly once in trajectory_segments.csv
    # Check 2: Total rows equals 4,945
    total_raw_rows = sum(len(rows) for rows in raw_obs_by_video.values())
    total_seg_rows = len(seg_obs)
    check_2_pass = (total_seg_rows == 4945 and total_raw_rows == 4945)
    
    # Check 1 verification: Unique keys (video, track_id, frame_id)
    obs_keys = [(r['video_name'], r['track_id'], r['frame_id']) for r in seg_obs]
    check_1_pass = (len(obs_keys) == len(set(obs_keys)) == 4945)
    
    results["check_1_observations_appear_once"] = {
        "status": "PASS" if check_1_pass else "FAIL",
        "description": "Every original observation appears exactly once without duplication in trajectory_segments.csv.",
        "unique_keys_count": len(set(obs_keys)),
        "total_rows": len(obs_keys)
    }
    
    results["check_2_row_count_equals_4945"] = {
        "status": "PASS" if check_2_pass else "FAIL",
        "description": "Number of rows in trajectory_segments.csv exactly equals 4,945.",
        "row_count": total_seg_rows
    }
    
    # Check 3: No original center coordinates were changed
    coords_match = True
    for v_key, raw_rows in raw_obs_by_video.items():
        raw_map = {(r['track_id'], int(r['frame_id'])): (float(r['center_x']), float(r['center_y'])) for r in raw_rows}
        v_seg_rows = [r for r in seg_obs if r['video_name'] == v_key]
        for s_r in v_seg_rows:
            key = (str(s_r['track_id']), s_r['frame_id'])
            raw_cx, raw_cy = raw_map[key]
            if raw_cx != s_r['x'] or raw_cy != s_r['y']:
                coords_match = False
                break
    results["check_3_center_coordinates_unchanged"] = {
        "status": "PASS" if coords_match else "FAIL",
        "description": "Center coordinates (x, y) exactly match original center_x and center_y with zero modification."
    }
    
    # Check 4: No original frame IDs were changed
    # Check 5: No original Track IDs were changed
    ids_match = True
    for v_key, raw_rows in raw_obs_by_video.items():
        raw_pairs = set((int(r['track_id']), int(r['frame_id'])) for r in raw_rows)
        seg_pairs = set((r['track_id'], r['frame_id']) for r in seg_obs if r['video_name'] == v_key)
        if raw_pairs != seg_pairs:
            ids_match = False
    results["check_4_frame_ids_unchanged"] = {
        "status": "PASS" if ids_match else "FAIL",
        "description": "All frame IDs strictly match original input tracking frames without alteration."
    }
    results["check_5_track_ids_unchanged"] = {
        "status": "PASS" if ids_match else "FAIL",
        "description": "All Track IDs strictly match original input Track IDs without alteration."
    }
    
    # Check 6: No synthetic/interpolated observations exist
    results["check_6_no_synthetic_observations"] = {
        "status": "PASS" if (total_seg_rows == total_raw_rows == 4945) else "FAIL",
        "description": "Zero synthetic, imputed, or interpolated observations added."
    }
    
    # Check 7: Every segment has monotonically increasing frame IDs
    # Check 8: Every segment contains only consecutive frame IDs (gap == 1)
    monotonic_pass = True
    consecutive_pass = True
    seg_groups = defaultdict(list)
    for r in seg_obs:
        seg_groups[r['segment_id']].append(r['frame_id'])
        
    for seg_id, f_list in seg_groups.items():
        if f_list != sorted(f_list):
            monotonic_pass = False
        for i in range(len(f_list) - 1):
            if f_list[i+1] - f_list[i] != 1:
                consecutive_pass = False
                break
                
    results["check_7_monotonically_increasing_frames"] = {
        "status": "PASS" if monotonic_pass else "FAIL",
        "description": "Every continuous segment has strictly monotonically increasing frame IDs."
    }
    results["check_8_only_consecutive_frame_ids"] = {
        "status": "PASS" if consecutive_pass else "FAIL",
        "description": "Every continuous segment strictly contains consecutive frame IDs (frame_gap == 1)."
    }
    
    # Check 9: Every segment boundary corresponds to a real frame gap > 1
    boundaries_valid = True
    for g in gap_summaries:
        if g['frame_gap'] <= 1 or (g['next_frame'] - g['previous_frame'] <= 1):
            boundaries_valid = False
    results["check_9_segment_boundaries_correspond_to_gap_gt_1"] = {
        "status": "PASS" if boundaries_valid else "FAIL",
        "description": "All recorded segment boundaries correspond to real tracking gaps where frame_gap > 1.",
        "total_gaps_verified": len(gap_summaries)
    }
    
    # Check 10: No NaN or infinite values introduced
    nan_inf_found = False
    for r in seg_obs:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    nan_inf_found = True
    for r in seg_summaries:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    nan_inf_found = True
    for r in gap_summaries:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    nan_inf_found = True
                    
    results["check_10_no_nan_or_infinite_values"] = {
        "status": "PASS" if not nan_inf_found else "FAIL",
        "description": "Zero NaN or infinite values present across preprocessed observations, segment summaries, and gap summaries."
    }
    
    # Check 11: Total observations across all segments equals 4,945
    sum_obs_segments = sum(s['num_observations'] for s in seg_summaries)
    check_11_pass = (sum_obs_segments == 4945)
    results["check_11_sum_observations_equals_4945"] = {
        "status": "PASS" if check_11_pass else "FAIL",
        "description": "The sum of observations across all segment summaries exactly equals 4,945.",
        "sum_observations": sum_obs_segments
    }
    
    all_passed = all(c['status'] == "PASS" for c in results.values())
    results["overall_validation_status"] = "ALL CHECKS PASSED" if all_passed else "VALIDATION FAILED"
    
    return results

if __name__ == "__main__":
    run_trajectory_preprocessing()

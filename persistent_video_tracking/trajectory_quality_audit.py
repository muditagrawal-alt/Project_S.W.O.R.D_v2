import os
import csv
import json
import math
from collections import defaultdict
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

OUTPUT_DIR = r"persistent_video_tracking\dataset\outputs\reports\trajectory"

def calculate_euclidean(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def run_quality_audit():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_video_gaps = {}
    all_gaps_flat = []
    
    all_jumps = []
    video_track_summaries = []
    
    extreme_tracks_inspection = {}
    
    # Process each video
    for v_key, config in VIDEO_CONFIGS.items():
        csv_path = config["input_csv"]
        fps = config["fps"]
        width = config["width"]
        height = config["height"]
        diag = math.hypot(width, height)
        large_jump_threshold = max(150.0, 0.10 * diag)
        
        with open(csv_path, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
            
        tracks = defaultdict(list)
        for row in reader:
            tracks[int(row['track_id'])].append({
                'frame_id': int(row['frame_id']),
                'center_x': float(row['center_x']),
                'center_y': float(row['center_y']),
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
                'confidence': float(row['confidence']),
                'class': row['class'],
                'smoothed_class': row['smoothed_class']
            })
            
        v_gaps = []
        
        sorted_track_ids = sorted(tracks.keys())
        for tid in sorted_track_ids:
            obs = tracks[tid]
            obs.sort(key=lambda x: x['frame_id'])
            
            num_det = len(obs)
            first_frame = obs[0]['frame_id']
            last_frame = obs[-1]['frame_id']
            duration_s = (last_frame - first_frame) / fps if fps > 0 else 0.0
            
            track_gaps = []
            track_jumps = []
            total_dist = 0.0
            
            for i in range(num_det - 1):
                p_prev = obs[i]
                p_curr = obs[i+1]
                
                gap = p_curr['frame_id'] - p_prev['frame_id']
                track_gaps.append(gap)
                v_gaps.append(gap)
                all_gaps_flat.append(gap)
                
                jump = calculate_euclidean(p_prev['center_x'], p_prev['center_y'],
                                           p_curr['center_x'], p_curr['center_y'])
                total_dist += jump
                
                jump_per_gap = jump / gap if gap > 0 else 0.0
                
                jump_record = {
                    'video': v_key,
                    'track_id': tid,
                    'previous_frame': p_prev['frame_id'],
                    'current_frame': p_curr['frame_id'],
                    'frame_gap': gap,
                    'previous_x': round(p_prev['center_x'], 2),
                    'previous_y': round(p_prev['center_y'], 2),
                    'current_x': round(p_curr['center_x'], 2),
                    'current_y': round(p_curr['center_y'], 2),
                    'pixel_jump': round(jump, 4),
                    'pixel_jump_per_frame_gap': round(jump_per_gap, 4),
                    'smoothed_class': p_curr['smoothed_class'],
                    'prev_bbox_wh': (round(p_prev['x2'] - p_prev['x1'], 1), round(p_prev['y2'] - p_prev['y1'], 1)),
                    'curr_bbox_wh': (round(p_curr['x2'] - p_curr['x1'], 1), round(p_curr['y2'] - p_curr['y1'], 1)),
                    'prev_conf': p_prev['confidence'],
                    'curr_conf': p_curr['confidence']
                }
                track_jumps.append(jump_record)
                all_jumps.append(jump_record)
                
            disp = calculate_euclidean(obs[0]['center_x'], obs[0]['center_y'],
                                       obs[-1]['center_x'], obs[-1]['center_y'])
            if total_dist > 0.0:
                straightness = min(1.0, disp / total_dist)
            else:
                straightness = 1.0 if disp == 0.0 else 0.0
                
            # Compute track level gap metrics
            if track_gaps:
                max_gap = max(track_gaps)
                mean_gap = round(float(np.mean(track_gaps)), 4)
                gaps_gt_1 = sum(1 for g in track_gaps if g > 1)
                gaps_gt_5 = sum(1 for g in track_gaps if g > 5)
                gaps_gt_10 = sum(1 for g in track_gaps if g > 10)
            else:
                max_gap = 0
                mean_gap = 0.0
                gaps_gt_1 = 0
                gaps_gt_5 = 0
                gaps_gt_10 = 0
                
            # Compute track level jump metrics
            if track_jumps:
                max_jump_rec = max(track_jumps, key=lambda x: x['pixel_jump'])
                max_jump = max_jump_rec['pixel_jump']
                max_jump_gap = max_jump_rec['frame_gap']
                max_jump_per_gap = max_jump_rec['pixel_jump_per_frame_gap']
                large_jumps_in_track = [j for j in track_jumps if j['pixel_jump'] > large_jump_threshold]
            else:
                max_jump = 0.0
                max_jump_gap = 0
                max_jump_per_gap = 0.0
                large_jumps_in_track = []
                
            # Quality flags
            flags = []
            if num_det == 1:
                flags.append("FLAG_A_SINGLE_DETECTION")
            elif num_det in (2, 3):
                flags.append("FLAG_B_FEW_DETECTIONS")
                
            if max_gap > 30:
                flags.append("FLAG_C_EXTREME_FRAME_GAP_GT30")
            elif max_gap > 10:
                flags.append("FLAG_C_LARGE_FRAME_GAP_GT10")
                
            if len(large_jumps_in_track) == 1:
                flags.append("FLAG_D_LARGE_SPATIAL_JUMP")
            elif len(large_jumps_in_track) > 1:
                flags.append("FLAG_E_REPEATED_LARGE_JUMPS")
                
            if num_det > 3 and straightness < 0.10:
                flags.append("FLAG_F_VERY_LOW_STRAIGHTNESS")
                
            if num_det > 3 and disp > 0 and (total_dist / disp) > 10.0:
                flags.append("FLAG_G_HIGH_DISTANCE_VS_DISPLACEMENT")
            elif num_det > 3 and disp == 0.0 and total_dist > 0:
                flags.append("FLAG_G_HIGH_DISTANCE_VS_DISPLACEMENT")
                
            quality_flags_str = ";".join(flags) if flags else "CLEAN_TRACK"
            
            video_track_summaries.append({
                'video_name': v_key,
                'track_id': tid,
                'num_detections': num_det,
                'first_frame': first_frame,
                'last_frame': last_frame,
                'duration_seconds': round(duration_s, 4),
                'max_frame_gap': max_gap,
                'mean_frame_gap': mean_gap,
                'num_gaps_gt_1': gaps_gt_1,
                'num_gaps_gt_5': gaps_gt_5,
                'num_gaps_gt_10': gaps_gt_10,
                'max_pixel_jump': round(max_jump, 4),
                'max_jump_frame_gap': max_jump_gap,
                'max_jump_per_frame_gap': round(max_jump_per_gap, 4),
                'total_distance_pixels': round(total_dist, 4),
                'displacement_pixels': round(disp, 4),
                'straightness_ratio': round(straightness, 4),
                'quality_flags': quality_flags_str
            })
            
        all_video_gaps[v_key] = v_gaps

    # Compute Gap Statistics per Video & Overall
    gap_stats = {}
    for v_key, gaps in all_video_gaps.items():
        if gaps:
            gap_stats[v_key] = {
                "total_transitions": len(gaps),
                "min_gap": int(min(gaps)),
                "max_gap": int(max(gaps)),
                "mean_gap": round(float(np.mean(gaps)), 4),
                "median_gap": float(np.median(gaps)),
                "num_gaps_gt_1": int(sum(1 for g in gaps if g > 1)),
                "pct_gaps_gt_1": round(float(sum(1 for g in gaps if g > 1) / len(gaps) * 100), 2),
                "num_gaps_gt_5": int(sum(1 for g in gaps if g > 5)),
                "pct_gaps_gt_5": round(float(sum(1 for g in gaps if g > 5) / len(gaps) * 100), 2),
                "num_gaps_gt_10": int(sum(1 for g in gaps if g > 10)),
                "pct_gaps_gt_10": round(float(sum(1 for g in gaps if g > 10) / len(gaps) * 100), 2),
                "num_gaps_gt_30": int(sum(1 for g in gaps if g > 30)),
                "pct_gaps_gt_30": round(float(sum(1 for g in gaps if g > 30) / len(gaps) * 100), 2),
            }
        else:
            gap_stats[v_key] = {"total_transitions": 0}
            
    gap_stats["overall"] = {
        "total_transitions": len(all_gaps_flat),
        "min_gap": int(min(all_gaps_flat)),
        "max_gap": int(max(all_gaps_flat)),
        "mean_gap": round(float(np.mean(all_gaps_flat)), 4),
        "median_gap": float(np.median(all_gaps_flat)),
        "num_gaps_gt_1": int(sum(1 for g in all_gaps_flat if g > 1)),
        "pct_gaps_gt_1": round(float(sum(1 for g in all_gaps_flat if g > 1) / len(all_gaps_flat) * 100), 2),
        "num_gaps_gt_5": int(sum(1 for g in all_gaps_flat if g > 5)),
        "pct_gaps_gt_5": round(float(sum(1 for g in all_gaps_flat if g > 5) / len(all_gaps_flat) * 100), 2),
        "num_gaps_gt_10": int(sum(1 for g in all_gaps_flat if g > 10)),
        "pct_gaps_gt_10": round(float(sum(1 for g in all_gaps_flat if g > 10) / len(all_gaps_flat) * 100), 2),
        "num_gaps_gt_30": int(sum(1 for g in all_gaps_flat if g > 30)),
        "pct_gaps_gt_30": round(float(sum(1 for g in all_gaps_flat if g > 30) / len(all_gaps_flat) * 100), 2),
    }

    # Top largest spatial jumps across all videos
    sorted_jumps_by_dist = sorted(all_jumps, key=lambda x: x['pixel_jump'], reverse=True)
    top_20_jumps = sorted_jumps_by_dist[:20]
    
    # Top largest jump per frame gap (highest apparent rate)
    sorted_by_jump_rate = sorted(all_jumps, key=lambda x: x['pixel_jump_per_frame_gap'], reverse=True)
    top_10_rates = sorted_by_jump_rate[:10]

    # Detailed Investigation on the 4 Extreme Tracks:
    # Video 3 Track #21, Video 3 Track #24, Video 3 Track #2, Video 2 Track #14
    target_extreme_tracks = [
        ("sample_video_03.mp4", 21),
        ("sample_video_03.mp4", 24),
        ("sample_video_03.mp4", 2),
        ("sample_video_02.mp4", 14)
    ]
    
    for v_target, t_target in target_extreme_tracks:
        t_summary = next(s for s in video_track_summaries if s['video_name'] == v_target and s['track_id'] == t_target)
        t_jumps = [j for j in all_jumps if j['video'] == v_target and j['track_id'] == t_target]
        top_t_jumps = sorted(t_jumps, key=lambda x: x['pixel_jump'], reverse=True)[:5]
        
        # Analyze evidence
        extreme_tracks_inspection[f"{v_target}_track_{t_target}"] = {
            "video_name": v_target,
            "track_id": t_target,
            "num_detections": t_summary['num_detections'],
            "first_frame": t_summary['first_frame'],
            "last_frame": t_summary['last_frame'],
            "duration_seconds": t_summary['duration_seconds'],
            "total_distance_pixels": t_summary['total_distance_pixels'],
            "displacement_pixels": t_summary['displacement_pixels'],
            "straightness_ratio": t_summary['straightness_ratio'],
            "max_frame_gap": t_summary['max_frame_gap'],
            "mean_frame_gap": t_summary['mean_frame_gap'],
            "max_pixel_jump": t_summary['max_pixel_jump'],
            "max_jump_frame_gap": t_summary['max_jump_frame_gap'],
            "max_jump_per_frame_gap": t_summary['max_jump_per_frame_gap'],
            "quality_flags": t_summary['quality_flags'],
            "top_5_largest_jumps": top_t_jumps
        }

    # Summary of quality flags across all 61 tracks
    flag_breakdown = defaultdict(list)
    for s in video_track_summaries:
        flags = s['quality_flags'].split(";")
        for f in flags:
            flag_breakdown[f].append({"video": s['video_name'], "track_id": s['track_id']})
            
    quality_report = {
        "report_title": "S.W.O.R.D Phase 2 Read-Only Trajectory Quality Audit",
        "description": "Comprehensive diagnostic audit of frame gaps, spatial jump magnitudes, continuity, and tracking flags across sample videos.",
        "note": "Descriptive diagnostic evaluation only. No velocity, acceleration, or threat scoring applied. Original observations preserved strictly without interpolation.",
        "dataset_scope": {
            "total_videos": len(VIDEO_CONFIGS),
            "total_tracks": len(video_track_summaries),
            "total_consecutive_transitions": len(all_gaps_flat)
        },
        "frame_gap_statistics": gap_stats,
        "top_largest_spatial_jumps": top_20_jumps,
        "top_highest_jump_rates_per_gap": top_10_rates,
        "quality_flags_summary": {
            k: {"count": len(v), "tracks": v} for k, v in flag_breakdown.items()
        },
        "extreme_tracks_investigation": extreme_tracks_inspection
    }
    
    # Save CSV summary
    summary_csv_path = os.path.join(OUTPUT_DIR, "trajectory_quality_summary.csv")
    fieldnames = [
        'video_name', 'track_id', 'num_detections', 'first_frame', 'last_frame',
        'duration_seconds', 'max_frame_gap', 'mean_frame_gap', 'num_gaps_gt_1',
        'num_gaps_gt_5', 'num_gaps_gt_10', 'max_pixel_jump', 'max_jump_frame_gap',
        'max_jump_per_frame_gap', 'total_distance_pixels', 'displacement_pixels',
        'straightness_ratio', 'quality_flags'
    ]
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(video_track_summaries)
    print(f"Saved trajectory quality summary CSV to: {summary_csv_path}")
    
    # Save JSON report
    report_json_path = os.path.join(OUTPUT_DIR, "trajectory_quality_report.json")
    with open(report_json_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    print(f"Saved trajectory quality report JSON to: {report_json_path}")
    
    return quality_report, video_track_summaries

if __name__ == "__main__":
    run_quality_audit()

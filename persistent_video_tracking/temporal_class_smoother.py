import os
import csv
import json
from collections import defaultdict, Counter

PHASE1_CLASSES = {
    0: 'Pistol',
    1: 'Shotgun',
    2: 'assault rifle',
    3: 'grenade',
    4: 'knife',
    5: 'sword',
    6: 'Sniper Dataset',
    7: 'SMG',
    8: 'rocket launcher',
    9: 'Machine gun'
}

def smooth_tracking_csv(input_csv_path, output_csv_path, window_size=20):
    """
    Applies temporal class smoothing independently for each track_id.
    - Preserves all original columns and their values.
    - Adds 'smoothed_class' column.
    - Uses recent class history (confidence-weighted sliding window of size `window_size`).
    - Does not invent any classes.
    """
    if not os.path.exists(input_csv_path):
        print(f"Warning: Input file does not exist: {input_csv_path}")
        return None
        
    with open(input_csv_path, 'r', newline='') as f:
        reader = list(csv.DictReader(f))
        
    if not reader:
        print(f"Warning: Input file is empty: {input_csv_path}")
        return None

    # Track-level history storage: track_id -> list of detection records
    track_histories = defaultdict(list)
    
    # Validation / Comparison metrics
    track_raw_series = defaultdict(list)
    track_smooth_series = defaultdict(list)
    track_raw_class_counts = defaultdict(Counter)
    track_smooth_class_counts = defaultdict(Counter)
    
    smoothed_rows = []
    
    for row in reader:
        t_id = row['track_id']
        raw_class = row['class']
        conf = float(row['confidence'])
        frame_id = int(row['frame_id'])
        
        # Verify class belongs to Phase 1 classes
        if raw_class not in PHASE1_CLASSES.values():
            print(f"Warning: Class '{raw_class}' not in Phase 1 classes!")

        # Update track history
        track_histories[t_id].append({
            'class': raw_class,
            'confidence': conf,
            'frame_id': frame_id
        })
        
        # Recent history window of size `window_size`
        recent_window = track_histories[t_id][-window_size:]
        
        if len(recent_window) <= 1:
            # Insufficient history: use the current frame's detected class directly
            smoothed_class = raw_class
        else:
            # Confidence-weighted majority vote across recent history
            class_scores = defaultdict(float)
            for item in recent_window:
                class_scores[item['class']] += item['confidence']
                
            # Class with highest accumulated confidence in recent history
            smoothed_class = max(class_scores.items(), key=lambda x: x[1])[0]
            
        # Create output row maintaining exact column order with smoothed_class added
        new_row = {
            'video_name': row['video_name'],
            'frame_id': row['frame_id'],
            'track_id': row['track_id'],
            'class': row['class'],
            'smoothed_class': smoothed_class,
            'confidence': row['confidence'],
            'x1': row['x1'],
            'y1': row['y1'],
            'x2': row['x2'],
            'y2': row['y2'],
            'center_x': row['center_x'],
            'center_y': row['center_y']
        }
        smoothed_rows.append(new_row)
        
        track_raw_series[t_id].append(raw_class)
        track_smooth_series[t_id].append(smoothed_class)
        track_raw_class_counts[t_id][raw_class] += 1
        track_smooth_class_counts[t_id][smoothed_class] += 1

    # Write smoothed CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        'video_name', 'frame_id', 'track_id', 'class', 'smoothed_class',
        'confidence', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y'
    ]
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(smoothed_rows)
        
    print(f"Successfully wrote {len(smoothed_rows)} rows to {output_csv_path}")
    
    # Calculate comparison statistics
    comparison_data = {
        "input_csv": input_csv_path,
        "output_csv": output_csv_path,
        "window_size": window_size,
        "total_detections": len(smoothed_rows),
        "total_tracks": len(track_histories),
        "total_raw_class_switches": 0,
        "total_smoothed_class_switches": 0,
        "tracks": {}
    }
    
    for t_id in sorted(track_raw_series.keys(), key=lambda x: int(x)):
        raw_list = track_raw_series[t_id]
        smooth_list = track_smooth_series[t_id]
        
        raw_switches = sum(1 for a, b in zip(raw_list[:-1], raw_list[1:]) if a != b)
        smooth_switches = sum(1 for a, b in zip(smooth_list[:-1], smooth_list[1:]) if a != b)
        
        comparison_data["total_raw_class_switches"] += raw_switches
        comparison_data["total_smoothed_class_switches"] += smooth_switches
        
        dominant_raw = track_raw_class_counts[t_id].most_common(1)[0][0]
        dominant_smooth = track_smooth_class_counts[t_id].most_common(1)[0][0]
        
        comparison_data["tracks"][t_id] = {
            "total_detections": len(raw_list),
            "raw_class_distribution": dict(track_raw_class_counts[t_id]),
            "smoothed_class_distribution": dict(track_smooth_class_counts[t_id]),
            "raw_class_switches": raw_switches,
            "smoothed_class_switches": smooth_switches,
            "jitter_reduction_percent": ((raw_switches - smooth_switches) / raw_switches * 100) if raw_switches > 0 else 0.0,
            "dominant_raw_class": dominant_raw,
            "dominant_smoothed_class": dominant_smooth
        }
        
    reduction_pct = 0.0
    if comparison_data["total_raw_class_switches"] > 0:
        reduction_pct = (
            (comparison_data["total_raw_class_switches"] - comparison_data["total_smoothed_class_switches"])
            / comparison_data["total_raw_class_switches"] * 100.0
        )
    comparison_data["overall_jitter_reduction_percent"] = reduction_pct
    
    return comparison_data

def main():
    targets = [
        (
            r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_01_tracking.csv",
            r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_01_tracking_smoothed.csv"
        ),
        (
            r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_02_tracking.csv",
            r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_02_tracking_smoothed.csv"
        ),
        (
            r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_03_tracking.csv",
            r"persistent_video_tracking\dataset\outputs\tracking_results\sample_video_03_tracking_smoothed.csv"
        )
    ]
    
    all_reports = {}
    for in_path, out_path in targets:
        v_name = os.path.basename(in_path)
        print(f"\nProcessing {v_name}...")
        report = smooth_tracking_csv(in_path, out_path, window_size=20)
        if report:
            all_reports[v_name] = report
            
    report_json_path = r"persistent_video_tracking\dataset\outputs\reports\temporal_class_smoothing_report.json"
    with open(report_json_path, 'w') as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nAll smoothing reports saved to: {report_json_path}")

if __name__ == "__main__":
    main()

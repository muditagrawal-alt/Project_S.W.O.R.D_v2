import cv2
import numpy as np
import os
import glob
import csv
import argparse
from ultralytics import YOLO
import supervision as sv

def process_videos(model_path, video_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    
    model = YOLO(model_path)
    
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    # We will write all results to one CSV or individual CSVs. Let's do one master CSV.
    csv_path = os.path.join(processed_dir, "tracking_results.csv")
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['video_name', 'frame_id', 'track_id', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for vid_path in videos:
            video_name = os.path.basename(vid_path)
            out_vid_path = os.path.join(processed_dir, video_name)
            print(f"Processing {video_name}...")
            
            tracker = sv.ByteTrack()
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_vid_path, fourcc, fps, (w, h))
            
            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                
                results = model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)
                
                labels = []
                for i in range(len(detections)):
                    xyxy = detections.xyxy[i]
                    mask = detections.mask[i] if detections.mask is not None else None
                    confidence = detections.confidence[i]
                    class_id = detections.class_id[i]
                    tracker_id = detections.tracker_id[i]
                    
                    x1, y1, x2, y2 = xyxy
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    
                    class_name = model.names[class_id]
                    
                    writer.writerow({
                        'video_name': video_name,
                        'frame_id': frame_id,
                        'track_id': tracker_id,
                        'class': class_name,
                        'confidence': round(float(confidence), 4),
                        'x1': round(float(x1), 2),
                        'y1': round(float(y1), 2),
                        'x2': round(float(x2), 2),
                        'y2': round(float(y2), 2),
                        'center_x': round(float(center_x), 2),
                        'center_y': round(float(center_y), 2)
                    })
                    
                    labels.append(f"#{tracker_id} {class_name} {confidence:.2f}")
                
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                
                out.write(annotated_frame)
                
            cap.release()
            out.release()
    print("Done processing all videos.")

from pathlib import Path

if __name__ == "__main__":
    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parent
    default_model = str(repo_root / "weights" / "yolo26m_best.pt")
    default_input = str(pkg_root / "dataset" / "raw" / "videos")
    default_output = str(pkg_root / "dataset" / "processed")

    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack Pipeline on a dataset")
    parser.add_argument("--model", type=str, default=default_model, help="Path to YOLO weights")
    parser.add_argument("--input_dir", type=str, default=default_input, help="Directory with input videos")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save CSV and tracked videos")
    
    args = parser.parse_args()
    process_videos(args.model, args.input_dir, args.output_dir)


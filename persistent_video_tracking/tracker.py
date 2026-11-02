import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

def main(source_video_path, target_video_path, model_path):
    # 1. Load YOLO model from Phase 1
    model = YOLO(model_path)
    
    # 2. Initialize ByteTrack object from supervision
    tracker = sv.ByteTrack()
    
    # 3. Initialize Annotators for drawing bounding boxes and labels
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # 4. Get video information for processing
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    
    # Callback function to process each frame
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        # Run YOLO inference
        results = model(frame, verbose=False)[0]
        
        # Convert YOLO results to supervision Detections
        detections = sv.Detections.from_ultralytics(results)
        
        # Pass detections to the tracker to get Track IDs
        detections = tracker.update_with_detections(detections)
        
        # Prepare labels with Class Name, Confidence, and Track ID
        labels = [
            f"#{tracker_id} {model.names[class_id]} {confidence:.2f}"
            for class_id, tracker_id, confidence 
            in zip(detections.class_id, detections.tracker_id, detections.confidence)
        ]
        
        # Annotate frame
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        
        return annotated_frame
    
    # 5. Process video and save
    print(f"Processing video: {source_video_path}")
    sv.process_video(
        source_path=source_video_path,
        target_path=target_video_path,
        callback=callback
    )
    print(f"Tracking complete. Output saved to: {target_video_path}")

if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path
    
    pkg_dir = Path(__file__).resolve().parent
    repo_dir = pkg_dir.parent
    default_model = str(repo_dir / "weights" / "yolo26m_best.pt")
    
    parser = argparse.ArgumentParser(description="Persistent Video Tracking ByteTrack Prototype")
    parser.add_argument("--source", type=str, default="sample_test_video.mp4", help="Path to input video")
    parser.add_argument("--target", type=str, default="output_tracked.mp4", help="Path to output video")
    parser.add_argument("--model", type=str, default=default_model, help="Path to YOLO model weights")
    
    args = parser.parse_args()
    
    source_path = args.source
    if not os.path.exists(source_path):
        candidate = pkg_dir / source_path
        if candidate.exists():
            source_path = str(candidate)

    if not os.path.exists(source_path):
        print(f"Error: Source video '{args.source}' not found. Please provide a valid input video.")
    else:
        main(source_path, args.target, args.model)


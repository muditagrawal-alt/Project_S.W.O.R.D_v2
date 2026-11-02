import cv2
import json
import os
import glob

from pathlib import Path

def audit_dataset():
    base_dir = Path(__file__).resolve().parent / "dataset" / "raw"
    video_dir = str(base_dir / "videos")
    anno_dir = str(base_dir / "annotations")

    
    videos = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    print(f"{'Video':<20} | {'Class':<12} | {'Res':<10} | {'FPS':<4} | {'Frames':<6} | {'Dur(s)':<6} | {'Anno Frames':<11} | {'Verification'}")
    print("-" * 105)
    
    for vid_path in videos:
        name = os.path.basename(vid_path).replace('.mp4', '')
        anno_path = os.path.join(anno_dir, f"{name}.json")
        
        # Open Video
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        duration = round(frames / fps, 2) if fps > 0 else 0
        
        # Open Annotations
        with open(anno_path, 'r') as f:
            data = json.load(f)
            
        category = data.get("category", "Unknown")
        anno_frames = len(data.get("frames", []))
        
        # Verify frame_id matches video limits
        verification = "PASS"
        for frame_data in data.get("frames", []):
            f_id = frame_data["frame_id"]
            if f_id < 1 or f_id > frames:
                verification = f"FAIL (Bad ID {f_id})"
                break
                
        print(f"{name:<20} | {category:<12} | {f'{w}x{h}':<10} | {fps:<4} | {frames:<6} | {duration:<6} | {anno_frames:<11} | {verification}")

if __name__ == "__main__":
    audit_dataset()

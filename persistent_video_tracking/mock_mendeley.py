import cv2
import numpy as np
import json
import os
import random

from pathlib import Path

def create_mock_dataset():
    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parent
    video_dir = str(pkg_root / "dataset" / "raw" / "videos")
    anno_dir = str(pkg_root / "dataset" / "raw" / "annotations")
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)
    
    # We load real weapon images from the dataset so YOLO can detect them
    img_path = str(repo_root / "dataset" / "raw" / "Weapons_Dataset_30k_10_Classes" / "10 Machine gun" / "all_data" / "all_images" / "machine_gun1.jpg")
    obj_img = cv2.imread(img_path)

    if obj_img is not None:
        obj_img = cv2.resize(obj_img, (150, 100))
        h_obj, w_obj = obj_img.shape[:2]
    else:
        # Fallback to white box if image not found
        obj_img = np.ones((100, 150, 3), dtype=np.uint8) * 255
        h_obj, w_obj = 100, 150
    
    cases = [
        ("Handgun_001", "Handgun", 30, 640, 480, 60),
        ("Handgun_002", "Handgun", 24, 1280, 720, 48),
        ("Machine_Gun_001", "Machine_Gun", 30, 1920, 1080, 90),
        ("Machine_Gun_002", "Machine_Gun", 30, 640, 480, 75),
        ("No_Gun_001", "No_Gun", 24, 1280, 720, 50)
    ]
    
    for (name, category, fps, w, h, frames) in cases:
        video_path = os.path.join(video_dir, f"{name}.mp4")
        json_path = os.path.join(anno_dir, f"{name}.json")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        annotations = []
        
        for i in range(frames):
            frame = np.ones((h, w, 3), dtype=np.uint8) * 100
            
            if category != "No_Gun":
                x = int((w - w_obj) * (i / frames))
                y = int((h - h_obj) * 0.5 + np.sin(i * 0.2) * 50)
                
                # Occlude/Skip drawing the weapon randomly (10% chance) to test ByteTrack persistence
                draw_weapon = random.random() > 0.1
                
                if draw_weapon:
                    frame[y:y+h_obj, x:x+w_obj] = obj_img
                    annotations.append({
                        "frame_id": i + 1,
                        "annotations": [
                            {
                                "class": category,
                                "bbox": [x, y, w_obj, h_obj]
                            }
                        ]
                    })
            
            out.write(frame)
            
        out.release()
        
        data = {
            "video_name": f"{name}.mp4",
            "category": category,
            "frames": annotations
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Generated {name}.mp4 and {name}.json")

if __name__ == "__main__":
    create_mock_dataset()

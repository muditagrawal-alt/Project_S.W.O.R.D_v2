import os
from ultralytics import YOLO

base_dir = "/Users/muditagrawal/Projects/SWORD_v2/Weapons_Dataset_30k_10_Classes"
model_path = "best.pt"

# Map folder names to their expected YOLO class IDs
folder_to_class = {
    "1 Pistol": 0,
    "2 Shotgun": 1,
    "3 assault rifle": 2,
    "4 grenade": 3,
    "5 knife": 4,
    "6 sword": 5,
    "7 Sniper Dataset": 6,
    "8 SMG": 7,
    "9 rocket launcher": 8,
    "10 Machine gun": 9
}

def auto_label_hard_images():
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'.")
        return
        
    print(f"Loading model for a Deep Pass...")
    model = YOLO(model_path)
    
    total_new_labels = 0
    total_images_processed = 0
    
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)
        if not os.path.isdir(class_path) or class_dir not in folder_to_class:
            continue
            
        expected_class_id = folder_to_class[class_dir]
        images_dir = os.path.join(class_path, "all_data", "all_images")
        labels_dir = os.path.join(class_path, "all_data", "all_labels")
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            images = [f for f in os.listdir(images_dir) if f != '.DS_Store']
            existing_labels = set([os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f != '.DS_Store' and f.endswith('.txt')])
            
            for img_filename in images:
                img_basename = os.path.splitext(img_filename)[0]
                
                # Only process images that STILL don't have labels
                if img_basename not in existing_labels:
                    img_path = os.path.join(images_dir, img_filename)
                    lbl_path = os.path.join(labels_dir, img_basename + ".txt")
                    
                    try:
                        # 1. Lower confidence to 0.05 (will guess on very blurry/small items)
                        # 2. augment=True enables Test-Time Augmentation (scales/flips image to find hidden objects)
                        results = model.predict(img_path, conf=0.05, augment=True, verbose=False)
                    except Exception as e:
                        continue
                        
                    total_images_processed += 1
                    boxes = results[0].boxes
                    
                    valid_boxes = []
                    # 3. Filter the results: ONLY keep the box if it matches the folder's expected class
                    # This prevents random background noise from being labeled as the wrong weapon.
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id == expected_class_id:
                            valid_boxes.append(box)
                    
                    if len(valid_boxes) > 0:
                        with open(lbl_path, 'w') as f:
                            for box in valid_boxes:
                                x_center, y_center, width, height = box.xywhn[0]
                                f.write(f"{expected_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        total_new_labels += 1
                        
                    if total_images_processed % 100 == 0:
                        print(f"Deep scanned {total_images_processed} stubborn images... extracted {total_new_labels} new labels.")

    print("\n--- DEEP PASS COMPLETE ---")
    print(f"Successfully squeezed out {total_new_labels} new label files from the remaining dataset!")

if __name__ == "__main__":
    auto_label_hard_images()

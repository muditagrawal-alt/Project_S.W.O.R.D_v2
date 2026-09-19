import os
from ultralytics import YOLO

base_dir = "/Users/muditagrawal/Projects/SWORD_v2/Weapons_Dataset_30k_10_Classes"
model_path = "best.pt"  # This is the file you download from Kaggle

def auto_label():
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'.")
        print("Please download 'best.pt' from your Kaggle notebook (Output -> weapons_detection/yolov8_weapons_run1/weights/best.pt) and place it in this folder.")
        return
        
    print(f"Loading custom AI model from {model_path}...")
    model = YOLO(model_path)
    
    total_new_labels = 0
    total_images_processed = 0
    
    # Iterate through each class directory
    for class_dir in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        images_dir = os.path.join(class_path, "all_data", "all_images")
        labels_dir = os.path.join(class_path, "all_data", "all_labels")
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            images = [f for f in os.listdir(images_dir) if f != '.DS_Store']
            existing_labels = set([os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f != '.DS_Store' and f.endswith('.txt')])
            
            for img_filename in images:
                img_basename = os.path.splitext(img_filename)[0]
                
                # If the image doesn't already have a human-annotated label, use the AI!
                if img_basename not in existing_labels:
                    img_path = os.path.join(images_dir, img_filename)
                    lbl_path = os.path.join(labels_dir, img_basename + ".txt")
                    
                    # Run inference (conf=0.25 ignores low-confidence guesses)
                    # verbose=False prevents the console from getting spammed with 25,000 lines
                    try:
                        results = model.predict(img_path, conf=0.25, verbose=False)
                    except Exception as e:
                        print(f"Skipping {img_filename} due to error: {e}")
                        continue
                        
                    total_images_processed += 1
                    
                    # Write the detected bounding boxes to a YOLO .txt file
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        with open(lbl_path, 'w') as f:
                            for box in boxes:
                                cls_id = int(box.cls[0])
                                x_center, y_center, width, height = box.xywhn[0] # xywh normalized
                                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        total_new_labels += 1
                        
                    # Progress update
                    if total_images_processed % 500 == 0:
                        print(f"Scanned {total_images_processed} unlabeled images... generated {total_new_labels} new AI labels so far.")

    print("\n--- AUTO-LABELING COMPLETE ---")
    print(f"Successfully generated {total_new_labels} new label files!")

if __name__ == "__main__":
    auto_label()

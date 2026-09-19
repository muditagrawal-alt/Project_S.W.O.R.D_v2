import os
from collections import defaultdict

base_dir = "/Users/muditagrawal/Projects/SWORD_v2/Weapons_Dataset_30k_10_Classes"

results = []
total_all_images = 0
total_all_labels = 0

for class_dir in sorted(os.listdir(base_dir)):
    class_path = os.path.join(base_dir, class_dir)
    if not os.path.isdir(class_path):
        continue
    
    images_dir = os.path.join(class_path, "all_data", "all_images")
    labels_dir = os.path.join(class_path, "all_data", "all_labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        continue
        
    images = [f for f in os.listdir(images_dir) if f != '.DS_Store']
    labels = [f for f in os.listdir(labels_dir) if f != '.DS_Store' and f.endswith('.txt')]
    
    total_all_images += len(images)
    total_all_labels += len(labels)
    
    # Read labels to find class distribution
    class_counts = defaultdict(int)
    empty_labels = 0
    
    for lbl_file in labels:
        lbl_path = os.path.join(labels_dir, lbl_file)
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                empty_labels += 1
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                    except ValueError:
                        pass
                    
    results.append({
        "folder": class_dir,
        "total_images": len(images),
        "labeled_images": len(labels),
        "unlabeled_images": len(images) - len(labels),
        "class_counts": dict(class_counts),
        "empty_labels": empty_labels
    })

print(f"=== OVERALL STATISTICS ===")
print(f"Total Images in Dataset: {total_all_images}")
print(f"Total Labeled Files: {total_all_labels}")
print(f"Remaining Unlabeled: {total_all_images - total_all_labels}")
print("==========================\n")

for r in results:
    print(f"Folder: {r['folder']}")
    print(f"  Images: {r['total_images']} | Labeled: {r['labeled_images']} | Unlabeled left: {r['unlabeled_images']}")
    # Sort class counts by most frequent
    sorted_classes = dict(sorted(r['class_counts'].items(), key=lambda item: item[1], reverse=True))
    print(f"  Class IDs found in bounding boxes: {sorted_classes}")
    if r['empty_labels'] > 0:
        print(f"  Empty label files: {r['empty_labels']}")
    print("-" * 40)

import os
from glob import glob

base_dir = "/Users/muditagrawal/Projects/SWORD_v2/Weapons_Dataset_30k_10_Classes"

total_images = 0
total_labels = 0
matching_pairs = 0
missing_images_for_labels = []

for class_dir in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_dir)
    if not os.path.isdir(class_path):
        continue
    
    all_data_dir = os.path.join(class_path, "all_data")
    images_dir = os.path.join(all_data_dir, "all_images")
    labels_dir = os.path.join(all_data_dir, "all_labels")
    
    if os.path.exists(images_dir) and os.path.exists(labels_dir):
        images = set([os.path.splitext(f)[0] for f in os.listdir(images_dir) if f != '.DS_Store'])
        labels = set([os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f != '.DS_Store' and f.endswith('.txt') and f != 'classes.txt'])
        
        total_images += len(images)
        total_labels += len(labels)
        
        for label in labels:
            if label in images:
                matching_pairs += 1
            else:
                missing_images_for_labels.append(os.path.join(labels_dir, label + ".txt"))
                
print(f"Total Images: {total_images}")
print(f"Total Labels: {total_labels}")
print(f"Matching Pairs (Label has corresponding Image): {matching_pairs}")
if missing_images_for_labels:
    print(f"Missing images for {len(missing_images_for_labels)} labels. First 10:")
    for m in missing_images_for_labels[:10]:
        print(m)
else:
    print("All labels have a matching image.")

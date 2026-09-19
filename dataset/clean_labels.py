import os

base_dir = "/Users/muditagrawal/Projects/SWORD_v2/Weapons_Dataset_30k_10_Classes"

folder_to_class = {
    "1 Pistol": 0, "2 Shotgun": 1, "3 assault rifle": 2, "4 grenade": 3,
    "5 knife": 4, "6 sword": 5, "7 Sniper Dataset": 6, "8 SMG": 7,
    "9 rocket launcher": 8, "10 Machine gun": 9
}

def clean_false_positives():
    total_removed = 0
    total_kept = 0

    for class_dir, expected_class_id in folder_to_class.items():
        labels_dir = os.path.join(base_dir, class_dir, "all_data", "all_labels")
        if not os.path.exists(labels_dir):
            continue

        for lbl_file in os.listdir(labels_dir):
            if not lbl_file.endswith(".txt") or lbl_file == "classes.txt":
                continue
                
            lbl_path = os.path.join(labels_dir, lbl_file)
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            
            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    try:
                        cls_id = int(parts[0])
                        if cls_id == expected_class_id:
                            valid_lines.append(line)
                        else:
                            total_removed += 1
                    except ValueError:
                        pass
                        
            # Rewrite the file with only the valid lines
            with open(lbl_path, "w") as f:
                for line in valid_lines:
                    f.write(line)
                    total_kept += 1

    print(f"--- CLEANUP COMPLETE ---")
    print(f"Removed {total_removed} mismatched bounding boxes (false positives).")
    print(f"Kept {total_kept} perfectly matched bounding boxes.")

if __name__ == "__main__":
    clean_false_positives()

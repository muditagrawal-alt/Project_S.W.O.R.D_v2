"""
Remove boxes whose class does not match the image's source folder — NON-DESTRUCTIVELY.

The original version rewrote every label file in place (2026-06-18). That deleted legitimate boxes in
multi-weapon scenes (e.g. a knife in a "Pistol" image became unlabelled background) and destroyed the record of
what the labels were before. This version never touches the source labels:

    * default is a dry run that only reports what would change;
    * --apply writes cleaned copies to --out-dir (mirroring the folder structure) and logs every removed box
      to <out-dir>/removed_boxes.csv so the decision can be reviewed.

Prefer re-annotating multi-class images over deleting their off-folder boxes.

Usage:
    python dataset/clean_labels.py                         # dry run
    python dataset/clean_labels.py --apply --out-dir dataset/cleaned_labels
"""

import argparse
import csv
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "dataset" / "raw" / "Weapons_Dataset_30k_10_Classes"

folder_to_class = {
    "1 Pistol": 0, "2 Shotgun": 1, "3 assault rifle": 2, "4 grenade": 3,
    "5 knife": 4, "6 sword": 5, "7 Sniper Dataset": 6, "8 SMG": 7,
    "9 rocket launcher": 8, "10 Machine gun": 9
}


def clean_false_positives(base_dir: Path, out_dir: Path, apply: bool):
    total_removed = total_kept = files_changed = 0
    removed_log = []

    for class_dir, expected_class_id in folder_to_class.items():
        labels_dir = base_dir / class_dir / "all_data" / "all_labels"
        if not labels_dir.exists():
            continue

        for lbl_file in sorted(os.listdir(labels_dir)):
            if not lbl_file.endswith(".txt") or lbl_file == "classes.txt":
                continue
            with open(labels_dir / lbl_file) as f:
                lines = f.readlines()

            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    continue
                if cls_id == expected_class_id:
                    valid_lines.append(line)
                    total_kept += 1
                else:
                    total_removed += 1
                    removed_log.append({"folder": class_dir, "label_file": lbl_file, "removed_line": line.strip()})

            if len(valid_lines) != len([ln for ln in lines if ln.strip()]):
                files_changed += 1
            if apply:
                dst = out_dir / class_dir / "all_data" / "all_labels" / lbl_file
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "w") as f:
                    f.writelines(valid_lines)

    print(f"--- {'CLEANUP WRITTEN TO ' + str(out_dir) if apply else 'DRY RUN (nothing written)'} ---")
    print(f"Boxes that do not match their folder's class: {total_removed} (in {files_changed} files)")
    print(f"Boxes kept: {total_kept}")
    if apply:
        with open(out_dir / "removed_boxes.csv", "w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=["folder", "label_file", "removed_line"])
            w.writeheader()
            w.writerows(removed_log)
        print(f"Every removed box is listed in {out_dir / 'removed_boxes.csv'} for review.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dataset" / "cleaned_labels")
    ap.add_argument("--apply", action="store_true", help="write cleaned copies to --out-dir (source labels are never modified)")
    args = ap.parse_args()
    if args.apply and args.out_dir.resolve() == args.base_dir.resolve():
        raise SystemExit("--out-dir must differ from --base-dir: this script never overwrites source labels.")
    clean_false_positives(args.base_dir, args.out_dir, args.apply)

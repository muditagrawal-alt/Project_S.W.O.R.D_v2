"""
Pseudo-label images that have no human label, with full provenance and without touching human labels.

The original version wrote model predictions straight into the human label folders (all_labels/), so after the
fact nobody can tell which labels are human and which are model-made — and those pseudo-labels ended up in the
validation split as "ground truth". This version:

    * writes pseudo-labels to a SEPARATE folder (--out-dir), mirroring the class-folder structure;
    * records every written file in <out-dir>/provenance.csv (model sha256, thresholds, TTA, time, box count);
    * is a dry run unless --apply is given.

Pseudo-labels must never be used for validation or test images, and should be human-reviewed before training.

Usage:
    python dataset/auto_label.py --model archive/best.pt                     # dry run
    python dataset/auto_label.py --model archive/best.pt --apply
    python dataset/auto_label.py --model archive/best.pt --conf 0.05 --augment --folder-class-only --apply  # "hard" pass
"""

import argparse
import csv
import hashlib
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "dataset" / "raw" / "Weapons_Dataset_30k_10_Classes"

folder_to_class = {
    "1 Pistol": 0, "2 Shotgun": 1, "3 assault rifle": 2, "4 grenade": 3,
    "5 knife": 4, "6 sword": 5, "7 Sniper Dataset": 6, "8 SMG": 7,
    "9 rocket launcher": 8, "10 Machine gun": 9
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auto_label(args):
    from ultralytics import YOLO

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    model = YOLO(str(args.model))
    model_hash = sha256(args.model)
    provenance = []
    scanned = labelled = 0

    for class_dir in sorted(os.listdir(args.base_dir)):
        images_dir = args.base_dir / class_dir / "all_data" / "all_images"
        labels_dir = args.base_dir / class_dir / "all_data" / "all_labels"
        if not images_dir.exists() or class_dir not in folder_to_class:
            continue
        expected = folder_to_class[class_dir]
        human = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")} if labels_dir.exists() else set()
        for img_filename in sorted(os.listdir(images_dir)):
            stem, ext = os.path.splitext(img_filename)
            if ext.lower() not in {".jpg", ".jpeg", ".png"} or stem in human:
                continue  # only images without a human label
            scanned += 1
            if not args.apply:
                continue
            r = model.predict(str(images_dir / img_filename), conf=args.conf, augment=args.augment, verbose=False)[0]
            lines = []
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if args.folder_class_only and cls_id != expected:
                    continue
                x, y, w, h = box.xywhn[0].tolist()
                lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            if not lines:
                continue
            dst = args.out_dir / class_dir / f"{stem}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("".join(lines))
            labelled += 1
            provenance.append({"folder": class_dir, "image": img_filename, "label_file": str(dst.relative_to(args.out_dir)),
                               "n_boxes": len(lines), "model": str(args.model), "model_sha256": model_hash,
                               "conf": args.conf, "augment_tta": args.augment, "folder_class_only": args.folder_class_only,
                               "created_at": datetime.now().isoformat(timespec="seconds"), "human_verified": False})

    if args.apply:
        prov = args.out_dir / "provenance.csv"
        new = not prov.exists()
        with open(prov, "a", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(provenance[0]) if provenance else ["folder"])
            if new:
                w.writeheader()
            w.writerows(provenance)
        print(f"Wrote {labelled} pseudo-label files for {scanned} unlabelled images to {args.out_dir} (provenance.csv updated).")
    else:
        print(f"DRY RUN: {scanned} images have no human label. Re-run with --apply to write pseudo-labels to {args.out_dir}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "archive" / "best.pt")
    ap.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "dataset" / "pseudo_labels")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--augment", action="store_true", help="test-time augmentation")
    ap.add_argument("--folder-class-only", action="store_true", help="keep only boxes of the folder's class")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if args.out_dir.resolve() == args.base_dir.resolve() or "all_labels" in args.out_dir.parts:
        raise SystemExit("--out-dir must be separate from the human label folders.")
    auto_label(args)

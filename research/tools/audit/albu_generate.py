"""Re-create the dataset's own augmentation pipeline (copied verbatim from the script found at
dataset/yolo_dataset/images/train/rocket_launcher877.py) to produce known (source, augmented) pairs.

Runs in an isolated environment that has albumentations (it needs opencv-python-headless, which would
clash with the project venv's opencv-python):
    <albu-env>/bin/python albu_generate.py <sources.txt> <out_dir> <copies_per_source> <seed>
"""

import csv
import random
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

augment = A.ReplayCompose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=20, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.GaussNoise(p=0.2),
    A.HueSaturationValue(p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.08), rotate=(-10, 10), p=0.4),
])


def main():
    sources = [ln.strip() for ln in Path(sys.argv[1]).read_text().splitlines() if ln.strip()]
    out = Path(sys.argv[2])
    copies, seed = int(sys.argv[3]), int(sys.argv[4])
    random.seed(seed)
    np.random.seed(seed)
    augment.set_random_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, src in enumerate(sources):
        img = cv2.imread(src)
        if img is None:
            continue
        for k in range(copies):
            dst = out / f"aug_{i:04d}_{k}.jpg"
            res = augment(image=img)
            cv2.imwrite(str(dst), res["image"])
            applied = [t["__class_fullname__"].split(".")[-1] for t in res["replay"]["transforms"] if t.get("applied")]
            rows.append({"source": src, "augmented": str(dst), "copy": k, "applied": "+".join(applied) or "none"})
    with open(out / "pairs.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["source", "augmented", "copy", "applied"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} augmented images for {len(sources)} sources")


if __name__ == "__main__":
    main()

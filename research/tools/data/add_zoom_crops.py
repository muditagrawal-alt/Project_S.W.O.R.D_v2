"""
Zoom-domain training data: a new dataset version = the CCTV fine-tuning set (symlinked, unchanged) + training-only
crops that show weapons the way person-zoom inference sees them (tracking/zoom.py: an enlarged crop around a person).

Positive crops: a square around a random ground-truth weapon, side = s * max(weapon w, h) with s ~ U(3, 10)
(at least 96 px), random offset with the weapon fully inside; every box with >= 60% of its area inside is kept
(clipped). Negative crops: random squares (96-400 px) from frames without weapons (no-gun videos, replay negatives).
Validation and test splits are untouched, so results stay comparable with the full-frame model.

    python research/tools/data/add_zoom_crops.py --src research/data/cctv_det --out research/data/cctv_det_zoom
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import cv2


def read_labels(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().split("\n"):
        p = line.split()
        if len(p) == 5:
            out.append((int(p[0]), *map(float, p[1:])))
    return out


def to_xyxy(lab, W, H):
    c, x, y, w, h = lab
    return c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1, help="positive crops per positive training image")
    ap.add_argument("--k-neg", type=int, default=1, help="negative crops per negative training image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src, out = Path(args.src).resolve(), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    rng = random.Random(args.seed)
    # 1) symlink the original set (all splits)
    n_link = 0
    for kind in ("images", "labels"):
        for split_dir in sorted((src / kind).iterdir()):
            if not split_dir.is_dir():
                continue
            dst = out / kind / split_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in split_dir.iterdir():
                if f.suffix in (".jpg", ".jpeg", ".png", ".txt"):
                    os.symlink(f.resolve(), dst / f.name)
                    n_link += 1
    # 2) zoom crops from the training split (CCTV sources only: mendeley_*, ucf_*)
    stats = {"positive_crops": 0, "negative_crops": 0, "skipped": 0}
    for img_path in sorted((src / "images" / "train").glob("*")):
        if not img_path.name.startswith(("mendeley_", "ucf_")):
            continue
        labs = read_labels(src / "labels" / "train" / (img_path.stem + ".txt"))
        img = cv2.imread(str(img_path.resolve()))
        if img is None:
            stats["skipped"] += 1
            continue
        H, W = img.shape[:2]
        boxes = [to_xyxy(l, W, H) for l in labs]
        jobs = [("pos", i) for i in range(args.k)] if boxes else [("neg", i) for i in range(args.k_neg)]
        for kind, i in jobs:
            if kind == "pos":
                c, x1, y1, x2, y2 = rng.choice(boxes)
                bw, bh = x2 - x1, y2 - y1
                side = int(min(min(W, H), max(96, rng.uniform(3, 10) * max(bw, bh))))
                if side < max(bw, bh) + 2:
                    stats["skipped"] += 1
                    continue
                cx1 = int(rng.uniform(max(0, x2 - side), min(x1, W - side)))
                cy1 = int(rng.uniform(max(0, y2 - side), min(y1, H - side)))
            else:
                side = int(min(min(W, H), rng.uniform(96, 400)))
                cx1, cy1 = rng.randint(0, W - side), rng.randint(0, H - side)
            crop = img[cy1:cy1 + side, cx1:cx1 + side]
            lines = []
            for c, x1, y1, x2, y2 in boxes:
                ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx1 + side), min(y2, cy1 + side)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) < 0.6 * (x2 - x1) * (y2 - y1):
                    continue
                lines.append(f"{c} {((ix1 + ix2) / 2 - cx1) / side:.6f} {((iy1 + iy2) / 2 - cy1) / side:.6f} "
                             f"{(ix2 - ix1) / side:.6f} {(iy2 - iy1) / side:.6f}")
            name = f"zoom{kind}{i}_{img_path.stem}"
            cv2.imwrite(str(out / "images" / "train" / f"{name}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / "train" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats["positive_crops" if kind == "pos" else "negative_crops"] += 1
    yaml = (src / "dataset.yaml").read_text().replace(str(src), str(out.resolve()))
    yaml = "\n".join(f"val_web: {(out / 'images' / 'val_web').resolve()}" if l.startswith("val_web:") else l
                     for l in yaml.splitlines()) + "\n"
    (out / "dataset.yaml").write_text(yaml.replace("# S.W.O.R.D. CCTV fine-tuning set", "# S.W.O.R.D. CCTV fine-tuning set + zoom crops"))
    (out / "manifest.json").write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                                                   "source": str(src), "symlinked_files": n_link, **stats}, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

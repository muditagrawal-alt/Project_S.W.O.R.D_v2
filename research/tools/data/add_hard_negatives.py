"""
Detector training round 3 data: the zoom-crop CCTV set (research/data/cctv_det_zoom, symlinked, unchanged) plus
HARD NEGATIVES mined from the training split only, where the current detector fires on something that is not a weapon:

  frames   Mendeley P1 + P3 no-gun frames on which the deployed detector (full frame @1280 + hand crops, final
           replay) has a detection >= --min-score; frames already in the set (every 20th) are skipped; empty labels
  crops    square crops around those false detections (side 3-10x the box, >= 96 px, the scale of the zoom-crop
           positives) on the same frames, and around false detections on the CCTV-Gun / UCF-Crime training frames
           (full frame @1280, no overlap with any true box); empty labels

Validation and test splits are untouched, so results stay comparable with earlier detectors.

    python research/tools/data/add_hard_negatives.py --out research/data/cctv_det_zoom_hn
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "research" / "data"
SRC = DATA / "cctv_det_zoom"
MENDELEY = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset"
REPLAY = PROJECT_ROOT / "research/runs/20261026-113535_final-replay-mendeley-zoomcropsdet-hands"
DETECTOR = PROJECT_ROOT / "research/runs/20261026-075952_detector-yolo26s-cctv-ft-zoomcrops/train/weights/best.pt"


def overlaps(box, gts) -> bool:
    return any(min(box[2], g[2]) > max(box[0], g[0]) and min(box[3], g[3]) > max(box[1], g[1]) for g in gts)


def neg_crop(img, box, rng, gts=()):
    """Square crop of side s * max(w, h), s ~ U(3, 10), >= 96 px, containing the false detection, no true box."""
    h, w = img.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    side = int(min(max(96, rng.uniform(3, 10) * max(bw, bh)), min(w, h)))
    x0 = int(np.clip(rng.uniform(box[2] - side, box[0]), 0, w - side))
    y0 = int(np.clip(rng.uniform(box[3] - side, box[1]), 0, h - side))
    win = (x0, y0, x0 + side, y0 + side)
    if overlaps(win, gts):
        return None
    return img[y0:y0 + side, x0:x0 + side]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--max-crops", type=int, default=2400)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    rng, random_ = np.random.default_rng(args.seed), random.Random(args.seed)
    # 1) the zoom-crop set, symlinked file by file (resolved targets), all splits
    n_link = 0
    for kind in ("images", "labels"):
        for split_dir in sorted((SRC / kind).iterdir()):
            if split_dir.is_dir():
                dst = out / kind / split_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in split_dir.iterdir():
                    os.symlink(f.resolve(), dst / f.name)
                    n_link += 1
    img_dir, lab_dir = out / "images" / "train", out / "labels" / "train"

    def save(name, img):
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lab_dir / f"{name}.txt").write_text("")

    # 2) Mendeley P1 + P3 no-gun frames with false detections (from the deployed detector's replay)
    cands = []   # (video folder, stride, recorded idx, src frame, false boxes)
    for fj in sorted(REPLAY.glob("videos/mendeley__No_Gun__*/frames.json")):
        doc = json.loads(fj.read_text())
        parts = doc["video"].split("/")
        if parts[2].split("_")[2] not in ("P1", "P3"):
            continue
        for f in doc["frames"]:
            boxes = [d["bbox"] for d in f["weapon_detections"] if d["score"] >= args.min_score]
            if boxes:
                cands.append((parts[2], doc["fps"], f["frame_idx"], boxes))
    random_.shuffle(cands)
    by_video = defaultdict(list)
    for c in cands:
        by_video[c[0]].append(c)
    n_frames = n_crops = 0
    picked = {v: sorted(cs, key=lambda c: c[2])[: max(1, args.max_frames // max(len(by_video), 1))] for v, cs in by_video.items()}
    for vname, cs in sorted(picked.items()):
        cap = cv2.VideoCapture(str(MENDELEY / "No_Gun" / vname / "video.mp4"))
        stride = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 25.0) / cs[0][1])))
        want = {c[2] * stride: c for c in cs}
        src = -1
        while want:
            ok, img = cap.read()
            src += 1
            if not ok:
                break
            c = want.pop(src, None)
            if c is None:
                continue
            if src % 20 and n_frames < args.max_frames:     # every 20th no-gun frame is already in the set
                save(f"hardneg_frame_{vname}_f{src:05d}", img)
                n_frames += 1
            for b in c[3]:
                crop = neg_crop(img, b, rng)
                if crop is not None and n_crops < args.max_crops // 2:
                    save(f"hardneg_crop_{vname}_f{src:05d}_{n_crops}", crop)
                    n_crops += 1
        cap.release()
    n_crops_mendeley = n_crops
    # 3) CCTV-Gun / UCF-Crime training frames: crops around false detections of the current detector
    from ultralytics import YOLO

    model = YOLO(str(DETECTOR))
    for p in sorted((DATA / "cctv_det" / "images" / "train").glob("ucf_*.jpg")):
        if n_crops >= args.max_crops:
            break
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts = []
        lab = DATA / "cctv_det" / "labels" / "train" / (p.stem + ".txt")
        for line in (lab.read_text().split("\n") if lab.exists() else []):
            q = line.split()
            if len(q) == 5:
                cx, cy, bw, bh = (float(v) for v in q[1:])
                gts.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        r = model.predict(img, imgsz=1280, conf=args.min_score, device=args.device, verbose=False)[0]
        for k, b in enumerate(r.boxes.xyxy.cpu().numpy()):
            if overlaps(b, gts):
                continue
            crop = neg_crop(img, b, rng, gts)
            if crop is not None:
                save(f"hardneg_crop_{p.stem}_{k}", crop)
                n_crops += 1
    yaml = (SRC / "dataset.yaml").read_text().replace(str(SRC.resolve()), str(out.resolve()))
    (out / "dataset.yaml").write_text(yaml.replace("+ zoom crops", "+ zoom crops + hard negatives"))
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "source": str(SRC),
                "replay": str(REPLAY.relative_to(PROJECT_ROOT)), "detector": str(DETECTOR.relative_to(PROJECT_ROOT)),
                "symlinked_files": n_link, "hard_negative_frames": n_frames, "hard_negative_crops_mendeley": n_crops_mendeley,
                "hard_negative_crops_ucf": n_crops - n_crops_mendeley}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""
Build the CCTV fine-tuning set for the S.W.O.R.D. weapon detectors (YOLO format, 10 S.W.O.R.D. classes).

Sources (fetched by fetch_cctv_datasets.py; nothing in dataset/ is modified — replay images are symlinked):
  mendeley  surveillance-style firearm videos (IP CCTV camera 640x480 + smartphones 1080p), per-frame boxes.
            Location-disjoint split, declared before training: train P1 (CCTV) + P3 (phone); val P2 (CCTV);
            test P4 (CCTV) + P5 (phone; includes subjects V3/V4 never seen in training).
            Gun videos: every `--stride`-th annotated frame. No_Gun videos (walking, running, phone, bottle, heavy
            object): every `--neg-stride`-th frame as hard negatives (empty label files).
  ucf       CCTV-Gun handgun boxes on real UCF-Crime robbery/shooting CCTV frames (320x240). CCTV-Gun's own split is
            per image (neighbouring frames of one video land in train and test); re-split here BY VIDEO, stratified
            by category, 70/15/15, seed 0.
  replay    a class-stratified sample of the audit's lineage-grouped S.W.O.R.D. train split, so fine-tuning does not
            erase the original domain and the 8 classes CCTV data does not cover; a sample of its val split forms
            `val_web` (forgetting check only — model selection uses the CCTV-only `val`).
The mock-attack CCTV set (USRT) is NOT used: it stays the external test set.

Class mapping: Mendeley Handgun -> Pistol, Mendeley Machine_Gun -> Assault Rifle (rifle-type long gun),
CCTV-Gun handgun -> Pistol; CCTV-Gun person boxes are not used (the detector has no person class).

    python research/tools/data/build_cctv_detector_set.py --out research/data/cctv_det
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from ensemble.classes import CLASS_NAMES  # noqa: E402

DATA = PROJECT_ROOT / "research" / "data"
MENDELEY = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset"
UCF_VIDEOS = DATA / "ucf_crime" / "extracted" / "UCF_Crimes" / "Videos"
CCTVGUN = DATA / "cctv_gun" / "data" / "ucf"
AUDIT_SPLIT = PROJECT_ROOT / "research" / "runs" / "20261022-dataset-audit-full" / "proposed" / "split_v3_grouped"
MENDELEY_SPLIT = {"P1": "train", "P3": "train", "P2": "val", "P4": "test", "P5": "test"}
MENDELEY_CLASS = {"Handgun": CLASS_NAMES.index("Pistol"), "Machine_Gun": CLASS_NAMES.index("Assault Rifle")}
PISTOL = CLASS_NAMES.index("Pistol")


def parse_folder(name: str) -> dict:
    code, cam, loc, subj, light, seq = name.split("_")[:6]
    kind = "aim" if code.startswith("PA") else "carry" if code.startswith("PC") else "no_gun"
    return {"code": code, "camera": cam, "location": loc, "subject": subj, "brightness": light, "seq": seq, "kind": kind}


def yolo_line(cls: int, x: float, y: float, w: float, h: float, W: int, H: int) -> str | None:
    x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def read_frames(path: Path, keep: set):
    """Sequential decode (frame 0 = first frame), yielding (index, image) for indices in `keep`."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    last = max(keep) if keep else -1
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if i in keep:
            yield i, img
        i += 1
    cap.release()


def write_sample(out: Path, split: str, name: str, img, lines: list) -> None:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def build_mendeley(out: Path, stride: int, neg_stride: int, frame_offset: int, stats: dict) -> list:
    meta = []
    for cls_dir in sorted(MENDELEY.iterdir()):
        if not cls_dir.is_dir():
            continue
        for vdir in sorted(cls_dir.iterdir()):
            info = parse_folder(vdir.name)
            split = MENDELEY_SPLIT[info["location"]]
            video = vdir / "video.mp4"
            lab_path = vdir / "label.json"
            boxes = defaultdict(list)
            W = H = None
            if lab_path.exists():
                lab = json.loads(lab_path.read_text())
                cats = {c["id"]: c["name"] for c in lab["categories"]}
                imgs = {im["id"]: im for im in lab["images"]}
                for a in lab["annotations"]:
                    im = imgs[a["image_id"]]
                    W, H = im["width"], im["height"]
                    boxes[a["image_id"] - frame_offset].append((cats[a["category_id"]], a["bbox"]))
            if cls_dir.name == "No_Gun":
                cap = cv2.VideoCapture(str(video))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                keep = set(range(0, n, neg_stride))
            else:
                annotated = sorted(boxes)
                keep = set(annotated[::stride])
            n_written = 0
            for idx, img in read_frames(video, keep):
                h, w = img.shape[:2]
                lines = []
                for cname, (x, y, bw, bh) in boxes.get(idx, []):
                    if W and (W, H) != (w, h):   # annotation size differs from decoded size: rescale
                        sx, sy = w / W, h / H
                        x, y, bw, bh = x * sx, y * sy, bw * sx, bh * sy
                    ln = yolo_line(MENDELEY_CLASS.get(cname, PISTOL), x, y, bw, bh, w, h)
                    if ln:
                        lines.append(ln)
                write_sample(out, split, f"mendeley_{vdir.name}_f{idx:05d}", img, lines)
                stats[(split, "mendeley", "positive" if lines else "negative")] += 1
                n_written += 1
            meta.append({**info, "class": cls_dir.name, "split": split, "video": str(video.relative_to(DATA)),
                         "frames_written": n_written})
    return meta


def build_ucf(out: Path, seed: int, stats: dict) -> list:
    frames = json.loads((CCTVGUN / "frames.json").read_text())
    ann = json.loads((CCTVGUN / "annotation_detection" / "annotations_all.json").read_text())
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    by_img = defaultdict(list)
    for a in ann["annotations"]:
        if cats[a["category_id"]] == "handgun":
            by_img[a["image_id"]].append(a["bbox"])
    vids = sorted(frames)
    by_cat = defaultdict(list)
    for v in vids:
        by_cat["".join(ch for ch in v if ch.isalpha())].append(v)
    rng = random.Random(seed)
    split_of = {}
    for cat, vs in sorted(by_cat.items()):
        vs = vs[:]
        rng.shuffle(vs)
        n_val, n_test = max(1, round(0.15 * len(vs))), max(1, round(0.15 * len(vs)))
        for i, v in enumerate(vs):
            split_of[v] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    meta = []
    for v in vids:
        cat = "".join(ch for ch in v if ch.isalpha())
        path = UCF_VIDEOS / cat / f"{v}_x264.mp4"
        pairs = {int(fidx): int(img_id) for img_id, fidx in frames[v]}
        if not path.exists():
            meta.append({"video": v, "category": cat, "split": split_of[v], "missing": True})
            continue
        n = 0
        for idx, img in read_frames(path, set(pairs)):
            h, w = img.shape[:2]
            lines = [ln for ln in (yolo_line(PISTOL, *b, w, h) for b in by_img.get(pairs[idx], [])) if ln]
            write_sample(out, split_of[v], f"ucf_{v}_f{idx:05d}", img, lines)
            stats[(split_of[v], "ucf", "positive" if lines else "negative")] += 1
            n += 1
        meta.append({"video": v, "category": cat, "split": split_of[v], "frames_written": n})
    return meta


def build_replay(out: Path, n_train: int, n_val: int, seed: int, stats: dict) -> dict:
    """Symlink a class-stratified sample of the lineage-grouped S.W.O.R.D. split (dataset/ is only read)."""
    picked = {}
    # replay validation images go to a separate split (val_web): model selection must use CCTV validation only,
    # otherwise best.pt favours the web domain the original detector was trained on
    for split, n, src in (("train", n_train, "train.txt"), ("val_web", n_val, "val.txt")):
        paths = [Path(p) for p in (AUDIT_SPLIT / src).read_text().split()]
        by_cls = defaultdict(list)
        for p in paths:
            lab = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
            first = lab.read_text().split()[:1] if lab.exists() else []
            by_cls[int(first[0]) if first else -1].append((p, lab))
        rng = random.Random(seed)
        per = max(1, n // max(1, len(by_cls)))
        chosen = []
        for c in sorted(by_cls):
            items = by_cls[c][:]
            rng.shuffle(items)
            chosen += items[:per]
        for p, lab in chosen:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            dst_img = out / "images" / split / f"replay_{p.name}"
            dst_lab = out / "labels" / split / f"replay_{p.stem}.txt"
            if not dst_img.exists():
                os.symlink(p, dst_img)
            if lab.exists() and not dst_lab.exists():
                os.symlink(lab, dst_lab)
            stats[(split, "replay", "image")] += 1
        picked[split] = [str(p) for p, _ in chosen]
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA / "cctv_det"))
    # sized for training on an Apple M4 (measured 6.5 s/iteration at 960 px, batch 16): neighbouring frames are
    # near-duplicates, so thinning costs little information
    ap.add_argument("--stride", type=int, default=10, help="every n-th annotated frame of a gun video")
    ap.add_argument("--neg-stride", type=int, default=20, help="every n-th frame of a No_Gun video")
    ap.add_argument("--frame-offset", type=int, default=1, help="annotation image_id minus decoded frame index")
    ap.add_argument("--replay-train", type=int, default=1500)
    ap.add_argument("--replay-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new --out")
    stats: Counter = Counter()
    m_meta = build_mendeley(out, args.stride, args.neg_stride, args.frame_offset, stats)
    u_meta = build_ucf(out, args.seed, stats)
    replay = build_replay(out, args.replay_train, args.replay_val, args.seed, stats)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out / "dataset.yaml").write_text(f"# S.W.O.R.D. CCTV fine-tuning set (build_cctv_detector_set.py)\npath: {out.resolve()}\n"
                                      f"train: images/train\nval: images/val\ntest: images/test\nval_web: {(out / 'images' / 'val_web').resolve()}\n"
                                      f"nc: {len(CLASS_NAMES)}\nnames:\n{names}\n")
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                "counts": {f"{s}/{src}/{k}": v for (s, src, k), v in sorted(stats.items())},
                "mendeley_split": MENDELEY_SPLIT, "class_mapping": {"mendeley": {k: CLASS_NAMES[v] for k, v in MENDELEY_CLASS.items()},
                                                                    "cctv_gun_ucf": {"handgun": "Pistol"}},
                "mendeley_videos": m_meta, "ucf_videos": u_meta, "replay": replay,
                "not_used": "USRT mock-attack set (external test set)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=1))


if __name__ == "__main__":
    main()

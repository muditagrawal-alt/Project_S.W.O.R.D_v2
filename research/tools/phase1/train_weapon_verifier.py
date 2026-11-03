"""
Train the second-stage weapon verifier (tracking/verifier.py) on TRAINING-split data only.

Crops (square, 2x the box, 112 px) come from:
  Mendeley P1 + P3 (train) / P2 (val): the final replay's detections (full frame @1280 + hand crops, the deployed
    detector) on every recorded frame, matched to the dataset's own boxes. Detections with IoU >= 0.3 to a true
    weapon are positives, IoU < 0.1 negatives (most negatives are people holding phones, bottles or bags in the
    no-gun clips, the detector's main in-domain confusion). True boxes are added as jittered positives.
  CCTV-Gun / UCF-Crime frames of the detector's train (train) and val (val) splits (research/data/cctv_det): the
    same detector run on the full frame, matched to the labels in the same way.
Never used: Mendeley P4/P5, UCF-Crime test videos, mock-attack (USRT), vtest.

Model: ImageNet ResNet-18, binary output, AdamW, flips + colour jitter, positive class re-weighted; the epoch with
the best validation ROC-AUC is kept. --install writes weights/weapon_verifier_v1.pt.

    python research/tools/phase1/train_weapon_verifier.py --out research/runs/<id>_weapon-verifier [--install]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import tracking.verifier as V  # noqa: E402
from tracking.verifier import build_model, to_tensor_batch  # noqa: E402

CROP = {"scale": V.CROP_SCALE, "size": V.CROP_SIZE, "arch": "resnet18"}   # set from --crop-scale / --crop-size / --arch


def crop_square(img, box, jitter=None):
    return V.crop_square(img, box, CROP["scale"], CROP["size"], jitter)

DATA = PROJECT_ROOT / "research" / "data"
MENDELEY = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset"
REPLAY = PROJECT_ROOT / "research/runs/20261026-113535_final-replay-mendeley-zoomcropsdet-hands"
DETECTOR = PROJECT_ROOT / "research/runs/20261026-075952_detector-yolo26s-cctv-ft-zoomcrops/train/weights/best.pt"
SPLIT = {"P1": "train", "P3": "train", "P2": "val"}


def iou(a, b) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def mendeley_boxes(vdir: Path) -> tuple:
    """{source frame: [xyxy]} from label.json (image_id = frame + 1) and the annotated image size."""
    lab = vdir / "label.json"
    if not lab.exists():
        return {}, None
    j = json.loads(lab.read_text())
    imgs = {im["id"]: im for im in j["images"]}
    out, size = defaultdict(list), None
    for a in j["annotations"]:
        im = imgs[a["image_id"]]
        size = (im["width"], im["height"])
        x, y, w, h = a["bbox"]
        out[a["image_id"] - 1].append([x, y, x + w, y + h])
    return out, size


def add_crops(store, img, dets, gts, rng, n_jitter, caps, counts):
    m = iou([d["bbox"] for d in dets], gts)
    for k, d in enumerate(dets):
        best = m[k].max() if m.shape[1] else 0.0
        label = 1 if best >= 0.3 else 0 if best < 0.1 else None
        if label is None or counts[label] >= caps[label]:
            continue
        store["x"].append(crop_square(img, d["bbox"]))
        store["y"].append(label)
        store["src"].append("det_pos" if label else "det_neg")
        counts[label] += 1
    for g in gts:
        for _ in range(n_jitter):
            if counts["gt"] >= caps["gt"]:
                return
            jit = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2))
            store["x"].append(crop_square(img, g, jitter=jit))
            store["y"].append(1)
            store["src"].append("gt")
            counts["gt"] += 1


def mine_mendeley(stores, rng, log):
    for fj in sorted(REPLAY.glob("videos/*/frames.json")):
        doc = json.loads(fj.read_text())
        parts = doc["video"].split("/")
        split = SPLIT.get(parts[2].split("_")[2])
        if split is None:
            continue
        vdir = MENDELEY / parts[1] / parts[2]
        gt, size = mendeley_boxes(vdir)
        cap = cv2.VideoCapture(str(vdir / "video.mp4"))
        stride = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or 25.0) / doc["fps"])))
        frames = {f["frame_idx"]: f for f in doc["frames"]}
        caps = {1: 40, 0: 300, "gt": 40} if split == "train" else {1: 20, 0: 150, "gt": 20}
        counts = {1: 0, 0: 0, "gt": 0}
        src, idx = -1, 0
        while True:
            ok, img = cap.read()
            src += 1
            if not ok:
                break
            if src % stride:
                continue
            f = frames.get(idx)
            idx += 1
            if f is None or (not f["weapon_detections"] and src not in gt):
                continue
            g = np.asarray(gt.get(src, []), float).reshape(-1, 4)
            if size and len(g) and size != (img.shape[1], img.shape[0]):
                sx, sy = img.shape[1] / size[0], img.shape[0] / size[1]   # annotation size differs from decoded size
                g = g * np.array([sx, sy, sx, sy])
            dets = [d for d in f["weapon_detections"] if d["score"] >= 0.1]
            add_crops(stores[split], img, dets, g, rng, 1 if idx % 3 == 0 else 0, caps, counts)
        cap.release()
    log(f"mendeley: train {len(stores['train']['y'])}, val {len(stores['val']['y'])} crops")


def mine_ucf(stores, rng, log, device):
    from ultralytics import YOLO

    model = YOLO(str(DETECTOR))
    for split in ("train", "val"):
        imgs = sorted((DATA / "cctv_det" / "images" / split).glob("ucf_*.jpg"))
        for p in imgs:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            g = []
            lab = DATA / "cctv_det" / "labels" / split / (p.stem + ".txt")
            for line in (lab.read_text().split("\n") if lab.exists() else []):
                q = line.split()
                if len(q) == 5:
                    cx, cy, bw, bh = (float(v) for v in q[1:])
                    g.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            r = model.predict(img, imgsz=1280, conf=0.1, device=device, verbose=False)[0]
            dets = [{"bbox": b.tolist(), "score": float(s)} for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
            add_crops(stores[split], img, dets, np.asarray(g, float).reshape(-1, 4), rng, 1, {1: 5, 0: 10, "gt": 3},
                      {1: 0, 0: 0, "gt": 0})
        log(f"ucf {split}: {len(imgs)} images -> {split} now {len(stores[split]['y'])} crops")


def train(stores, out: Path, device: str, epochs: int, log) -> dict:
    import torch
    import torch.nn as nn
    from sklearn.metrics import average_precision_score, roc_auc_score

    xtr, ytr = np.stack(stores["train"]["x"]), np.asarray(stores["train"]["y"], np.float32)
    xva, yva = np.stack(stores["val"]["x"]), np.asarray(stores["val"]["y"], np.float32)
    model = build_model(CROP["arch"], pretrained=True)
    model.to(device)
    pos_w = torch.tensor([float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)

    def predict(x):
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                ps.append(torch.sigmoid(model(to_tensor_batch(x[i:i + 256]).to(device))).cpu().numpy().reshape(-1))
        return np.concatenate(ps)

    best, history = {"auc": -1}, []
    for ep in range(epochs):
        model.train()
        order, t0, tot = rng.permutation(len(xtr)), time.time(), 0.0
        for i in range(0, len(order), 128):
            b = order[i:i + 128]
            x = xtr[b].copy()
            flip = rng.random(len(b)) < 0.5
            x[flip] = x[flip, :, ::-1]
            x = np.clip(x.astype(np.float32) * rng.uniform(0.7, 1.3, (len(b), 1, 1, 1)) + rng.uniform(-25, 25, (len(b), 1, 1, 1)), 0, 255).astype(np.uint8)
            logits = model(to_tensor_batch(x).to(device)).reshape(-1)
            loss = loss_fn(logits, torch.from_numpy(ytr[b]).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        sched.step()
        p = predict(xva)
        auc, ap = roc_auc_score(yva, p), average_precision_score(yva, p)
        history.append({"epoch": ep + 1, "train_loss": round(tot / len(xtr), 4), "val_auc": round(auc, 4), "val_ap": round(ap, 4),
                        "seconds": round(time.time() - t0, 1)})
        log(json.dumps(history[-1]))
        if auc > best["auc"]:
            best = {"auc": auc, "ap": ap, "epoch": ep + 1}
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": CROP["arch"],
                        "crop_scale": CROP["scale"],
                        "crop_size": CROP["size"], "epoch": ep + 1, "val_auc": round(auc, 4), "val_ap": round(ap, 4),
                        "created": datetime.now().isoformat(timespec="seconds")}, out / "weapon_verifier.pt")
            np.save(out / "val_scores.npy", np.c_[yva, p])
    return {"best": best, "history": history, "train": {"n": len(ytr), "pos": int(ytr.sum())},
            "val": {"n": len(yva), "pos": int(yva.sum())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--crop-scale", type=float, default=V.CROP_SCALE, help="crop side / box side (context)")
    ap.add_argument("--crop-size", type=int, default=V.CROP_SIZE, help="crop size in pixels")
    ap.add_argument("--install-name", default="weapon_verifier_v1.pt")
    ap.add_argument("--arch", default="resnet18", choices=V.ARCHS)
    ap.add_argument("--crops-from", help="reuse crops.npz of an earlier run (same crop scale and size)")
    ap.add_argument("--replay", help="Mendeley replay to mine (default: the zoom-crop detector's final replay)")
    ap.add_argument("--detector", help="detector for the UCF frames (default: the zoom-crop detector)")
    args = ap.parse_args()
    CROP.update(scale=args.crop_scale, size=args.crop_size, arch=args.arch)
    global REPLAY, DETECTOR
    REPLAY = Path(args.replay).resolve() if args.replay else REPLAY
    DETECTOR = Path(args.detector).resolve() if args.detector else DETECTOR
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fp = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    random.seed(0)
    rng = np.random.default_rng(0)
    stores = {s: {"x": [], "y": [], "src": []} for s in ("train", "val")}
    cache = Path(args.crops_from) if args.crops_from else out / "crops.npz"
    if cache.exists():
        z = np.load(cache)
        for s in ("train", "val"):
            stores[s] = {"x": list(z[f"x_{s}"]), "y": list(z[f"y_{s}"]), "src": list(z[f"src_{s}"])}
    else:
        mine_mendeley(stores, rng, log)
        mine_ucf(stores, rng, log, args.device)
        np.savez_compressed(cache, **{f"{k}_{s}": np.asarray(stores[s][k]) for s in stores for k in ("x", "y", "src")})
    counts = {s: {src: int(sum(1 for q in stores[s]["src"] if q == src)) for src in ("gt", "det_pos", "det_neg")} for s in stores}
    log(f"crops: {json.dumps(counts)}")
    res = train(stores, out, args.device, args.epochs, log)
    result = {"created": datetime.now().isoformat(timespec="seconds"), "description":
              f"second-stage weapon verifier ({CROP['arch']}, {CROP['size']} px crops, {CROP['scale']}x context) trained on "
              "training-split crops only", "crop": CROP,
              "replay": str(REPLAY.relative_to(PROJECT_ROOT)), "detector": str(DETECTOR.relative_to(PROJECT_ROOT)),
              "crop_counts": counts, **res}
    (out / "results.json").write_text(json.dumps(result, indent=2))
    if args.install:
        dst = PROJECT_ROOT / "weights" / args.install_name
        shutil.copy2(out / "weapon_verifier.pt", dst)
        log(f"installed {dst}")


if __name__ == "__main__":
    main()

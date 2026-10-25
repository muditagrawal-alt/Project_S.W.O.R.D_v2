"""
SWORD v2 — corrected detection benchmark (replaces the protocol of evaluate_ensemble.py / run_sota_benchmarks.py).

What changed versus the legacy scripts, and why:
    * mAP from ALL predictions (conf >= 0.001) on the FULL evaluation list — not a 500-image sample with
      predictions cut at 0.20/0.15 (truncation caps recall; different cuts per configuration bias comparisons).
    * Ensemble parameters and the deployment threshold are chosen on a separate --tune-list, then applied
      unchanged to --test-list (the legacy optimiser tuned on the very images it reported on).
    * COCO AP / AP_small / AP_medium / AP_large from an independent evaluator; ranges with no objects are
      reported as n/a instead of 0.
    * Paired bootstrap confidence intervals; batch-1 latency on preloaded images.
    * WBF follows the reference implementation (ensemble/wbf.py, checked by tests/test_wbf.py).

Usage:
    # recommended: lineage-grouped split from the dataset audit (tune on val, report on test)
    python ensemble/benchmark.py --tune-list <val.txt> --test-list <test.txt>
    # leakage-free subset of the current val split (still the split used for checkpoint selection)
    python ensemble/benchmark.py --test-list research/runs/<audit>/proposed/clean_val_subset.txt
    # reproduce the legacy protocol for comparison
    python ensemble/benchmark.py --legacy-protocol
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.classes import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from ensemble.metrics import (best_threshold, bootstrap_ci, coco_metrics, detection_metrics,  # noqa: E402
                              label_path_for, load_yolo_labels, operating_point, precompute_tp)
from ensemble.wbf import nms, weighted_boxes_fusion  # noqa: E402

MODELS = {
    "YOLO26s": ("yolo", PROJECT_ROOT / "weights" / "yolo26s_best.pt"),
    "YOLO26m": ("yolo", PROJECT_ROOT / "weights" / "yolo26m_best.pt"),
    "RT-DETR-L": ("rtdetr", PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"),
}
DEFAULT_WBF = {"iou_thr": 0.55, "conf_type": "avg", "skip_box_thr": 0.001, "pre_nms_iou": 0.5}
# pre_nms_iou: class-wise NMS on each model's own output before fusion. Reference WBF averages scores inside a
# cluster and assumes ~one box per model per object; RT-DETR returns all 300 queries and YOLO26 (NMS-free) ~30
# overlapping boxes at conf 0.001, which otherwise dilute cluster scores (3-model mAP50 0.69 -> 0.89 on a smoke set).
MAX_BOXES_PER_MODEL = 100  # COCO maxDets; also bounds WBF cost
VAL_DIR = PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val"


# ───────────────────────────── inputs ─────────────────────────────
def read_list(path: str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def legacy_stratified_list(per_class: int = 50, seed: int = 42) -> list[str]:
    """The exact image list used by the legacy scripts (first label line decides the class; empty labels skipped)."""
    lbl_dir = str(VAL_DIR).replace("images", "labels")
    by_class = defaultdict(list)
    for f in sorted(os.listdir(lbl_dir)):
        p = os.path.join(lbl_dir, f)
        if not f.endswith(".txt") or os.path.getsize(p) == 0:
            continue
        with open(p) as fp:
            first = next((ln.split() for ln in fp if ln.strip()), None)
        if first:
            img = os.path.splitext(f)[0] + ".jpg"
            if not os.path.exists(os.path.join(VAL_DIR, img)):
                img = os.path.splitext(f)[0] + ".jpeg"
            by_class[int(first[0])].append(str(VAL_DIR / img))
    random.seed(seed)
    chosen = set()
    for c in range(NUM_CLASSES):
        chosen.update(random.sample(by_class[c], min(per_class, len(by_class[c]))))
    return sorted(chosen)


def list_digest(paths: list[str]) -> str:
    return hashlib.sha1("\n".join(paths).encode()).hexdigest()[:10]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


# ───────────────────────────── predictions ─────────────────────────────
def load_model(kind: str, weights: Path):
    from ultralytics import RTDETR, YOLO

    return RTDETR(str(weights)) if kind == "rtdetr" else YOLO(str(weights))


def predict_records(name: str, paths: list[str], device: str, cache_dir: Path, log) -> list[dict]:
    """Raw predictions at conf=0.001 for every image, cached per (weights hash, image list)."""
    import pickle

    kind, weights = MODELS[name]
    key = f"{name}_{file_digest(weights)}_{list_digest(paths)}.pkl"
    cache = cache_dir / key
    if cache.exists():
        log(f"  [cache] {name} predictions")
        return pickle.loads(cache.read_bytes())
    model = load_model(kind, weights)
    recs = []
    t0 = time.time()
    for s in range(0, len(paths), 16):
        batch = paths[s:s + 16]
        for img_id, (p, r) in enumerate(zip(batch, model.predict(batch, imgsz=640, conf=0.001, iou=0.7, max_det=300,
                                                                   device=device, verbose=False)), start=s):
            h, w = r.orig_shape
            gt_cls, gt_boxes = load_yolo_labels(label_path_for(p), w, h)
            b = r.boxes
            recs.append({"image_id": img_id, "path": p, "width": int(w), "height": int(h),
                         "gt_cls": gt_cls, "gt_boxes": gt_boxes,
                         "pred_cls": b.cls.cpu().numpy().astype(np.int64),
                         "pred_scores": b.conf.cpu().numpy().astype(np.float64),
                         "pred_boxes": b.xyxy.cpu().numpy().astype(np.float64)})
    log(f"  {name}: {len(paths)} images in {time.time() - t0:.0f}s")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(recs))
    return recs


def truncate(records: list[dict], thr: float) -> list[dict]:
    out = []
    for r in records:
        k = r["pred_scores"] >= thr
        out.append({**r, "pred_cls": r["pred_cls"][k], "pred_scores": r["pred_scores"][k], "pred_boxes": r["pred_boxes"][k]})
    return out


def fuse(per_model: list[list[dict]], weights: list[float], iou_thr: float, conf_type: str, skip_box_thr: float,
         pre_nms_iou: float | None = 0.5) -> list[dict]:
    fused = []
    for recs in zip(*per_model):
        w, h = recs[0]["width"], recs[0]["height"]
        scale = np.array([w, h, w, h], dtype=np.float64)
        boxes, scores, labels = [], [], []
        for r in recs:
            b, sc, c = r["pred_boxes"], r["pred_scores"], r["pred_cls"]
            if pre_nms_iou:
                b, sc, c = nms(b, sc, c, iou_thr=pre_nms_iou)
            top = np.argsort(-sc)[:MAX_BOXES_PER_MODEL]
            boxes.append(b[top] / scale)
            scores.append(sc[top])
            labels.append(c[top])
        fb, fs, fl = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                           skip_box_thr=skip_box_thr, conf_type=conf_type)
        fused.append({**recs[0], "pred_boxes": fb.astype(np.float64) * scale, "pred_scores": fs.astype(np.float64),
                      "pred_cls": fl.astype(np.int64)})
    return fused


def tune_wbf(per_model: list[list[dict]], log) -> dict:
    """Small grid on the TUNING split only; objective mAP50-95."""
    n = len(per_model)
    weight_grid = [[1.0] * n, [1.0, 1.1, 1.1][:n], [1.0, 1.0, 1.5][:n], [1.5, 1.0, 1.0][:n]]
    best = None
    for pre in (0.5, 0.6):
        for skip in (0.001, 0.05):
            for w in weight_grid:
                for iou in (0.5, 0.55, 0.6, 0.7):
                    for ct in ("avg", "box_and_model_avg"):
                        m = detection_metrics(fuse(per_model, w, iou, ct, skip, pre), CLASS_NAMES)["mAP50-95"]
                        if best is None or m > best[0]:
                            best = (m, {"weights": w, "iou_thr": iou, "conf_type": ct, "skip_box_thr": skip, "pre_nms_iou": pre})
    log(f"  tuned WBF ({n} models): {best[1]} -> tune mAP50-95 {best[0]:.4f}")
    return {**best[1], "tune_mAP50-95": best[0]}


# ───────────────────────────── latency ─────────────────────────────
def latency(names: list[str], paths: list[str], device: str, n: int, wbf_cfgs: dict, log) -> dict:
    import cv2

    imgs = [cv2.imread(p) for p in paths[:n]]
    models = {k: load_model(*MODELS[k]) for k in names}
    out = {}
    raw = {}
    for k, m in models.items():
        for im in imgs[:10]:
            m.predict(im, imgsz=640, conf=0.001, device=device, verbose=False)
        ts, dets = [], []
        for im in imgs:
            t0 = time.perf_counter()
            r = m.predict(im, imgsz=640, conf=0.001, device=device, verbose=False)[0]
            ts.append((time.perf_counter() - t0) * 1000)
            dets.append(r)
        raw[k] = (ts, dets)
        out[k] = {"ms_median": float(np.median(ts)), "ms_p95": float(np.percentile(ts, 95)), "fps_median": float(1000 / np.median(ts))}
    for cfg_name, (members, cfg) in wbf_cfgs.items():
        ts = []
        for i in range(len(imgs)):
            t_models = sum(raw[k][0][i] for k in members)
            r0 = raw[members[0]][1][i]
            h, w = r0.orig_shape
            scale = np.array([w, h, w, h])
            t0 = time.perf_counter()
            per = [nms(raw[k][1][i].boxes.xyxy.cpu().numpy(), raw[k][1][i].boxes.conf.cpu().numpy(),
                       raw[k][1][i].boxes.cls.cpu().numpy(), iou_thr=cfg.get("pre_nms_iou", 0.5)) for k in members]
            weighted_boxes_fusion([b[:MAX_BOXES_PER_MODEL] / scale for b, _, _ in per], [sc[:MAX_BOXES_PER_MODEL] for _, sc, _ in per],
                                  [c[:MAX_BOXES_PER_MODEL] for _, _, c in per],
                                  weights=cfg["weights"], iou_thr=cfg["iou_thr"], conf_type=cfg["conf_type"])
            ts.append(t_models + (time.perf_counter() - t0) * 1000)
        out[cfg_name] = {"ms_median": float(np.median(ts)), "ms_p95": float(np.percentile(ts, 95)),
                         "fps_median": float(1000 / np.median(ts)), "note": "sequential models + WBF"}
    log("  latency (batch 1): " + ", ".join(f"{k} {v['ms_median']:.1f} ms" for k, v in out.items()))
    return out


# ───────────────────────────── bookkeeping ─────────────────────────────
def git(cmd: list[str]) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def manifest(args, run_dir: Path, lists: dict) -> dict:
    from importlib import metadata

    pk = {}
    for p in ("numpy", "torch", "ultralytics", "opencv-python", "faster-coco-eval"):
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    import torch

    return {"run_id": run_dir.name, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv), "args": vars(args),
            "git": {"commit": git(["rev-parse", "HEAD"]), "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
                    "dirty": bool(git(["status", "--porcelain"]))},
            "hardware": {"platform": platform.platform(), "python": sys.version.split()[0],
                         "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()},
            "packages": pk,
            "weights_sha256_12": {k: file_digest(v[1]) for k, v in MODELS.items() if v[1].exists()},
            "image_lists": {k: {"n": len(v), "sha1_10": list_digest(v)} for k, v in lists.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-list")
    ap.add_argument("--tune-list")
    ap.add_argument("--legacy-protocol", action="store_true")
    ap.add_argument("--models", default="YOLO26s,YOLO26m,RT-DETR-L")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--latency-images", type=int, default=100)
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "research" / "runs"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    names = [m.strip() for m in args.models.split(",")]
    run_name = args.run_name or ("phase1-benchmark-legacy" if args.legacy_protocol else "phase1-benchmark")
    run_dir = Path(args.out_root) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(run_dir / "benchmark.log", "a")

    def log(msg: str):
        line = f"{datetime.now().strftime('%H:%M:%S')} | {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")

    if args.legacy_protocol:
        test = legacy_stratified_list()
        tune = []
    else:
        test = read_list(args.test_list) if args.test_list else sorted(str(p) for p in VAL_DIR.glob("*.jp*g"))
        tune = read_list(args.tune_list) if args.tune_list else []
    if args.max_images:
        rng = random.Random(args.seed)
        test = sorted(rng.sample(test, min(args.max_images, len(test))))
        tune = sorted(rng.sample(tune, min(args.max_images, len(tune)))) if tune else []
    warnings = []
    if not args.legacy_protocol and not args.test_list:
        warnings.append("Test list = full current val split, which was also used to select best.pt and contains "
                        "near-duplicates of train images: treat the numbers as optimistic.")
    if not tune and not args.legacy_protocol:
        warnings.append("No --tune-list: ensembles use default WBF parameters and the operating threshold is fixed at 0.25.")
    (run_dir / "manifest.json").write_text(json.dumps(manifest(args, run_dir, {"test": test, "tune": tune}), indent=2))
    log(f"Run: {run_dir}\n  device={device} test={len(test)} tune={len(tune)} legacy={args.legacy_protocol}")

    cache_dir = PROJECT_ROOT / "research" / "cache" / "predictions"
    test_recs = {n: predict_records(n, test, device, cache_dir, log) for n in names}
    tune_recs = {n: predict_records(n, tune, device, cache_dir, log) for n in names} if tune else {}

    ens_members = {f"WBF({'+'.join(names[:2])})": names[:2]}
    if len(names) >= 3:
        ens_members[f"WBF({'+'.join(names[:3])})"] = names[:3]
    wbf_cfg = {}
    for en, members in ens_members.items():
        default = {**DEFAULT_WBF, "weights": [1.0, 1.1, 1.1][:len(members)]}
        wbf_cfg[en] = tune_wbf([tune_recs[m] for m in members], log) if tune_recs else default

    configs = {n: test_recs[n] for n in names}
    for en, members in ens_members.items():
        c = wbf_cfg[en]
        configs[en] = fuse([test_recs[m] for m in members], c["weights"], c["iou_thr"], c["conf_type"], c["skip_box_thr"],
                           c.get("pre_nms_iou", 0.5))

    results = {}
    for name, recs in configs.items():
        if args.legacy_protocol:  # reproduce the legacy cut-offs exactly
            recs = truncate(recs, 0.15 if name.startswith("WBF") and name.count("+") == 2 else 0.20)
        precompute_tp(recs)
        res = {"detection": detection_metrics(recs, CLASS_NAMES)}
        if not args.legacy_protocol:
            res["coco"] = coco_metrics(recs, CLASS_NAMES)
            if tune_recs:
                tune_c = tune_recs[name] if name in tune_recs else fuse([tune_recs[m] for m in ens_members[name]],
                                                                        wbf_cfg[name]["weights"], wbf_cfg[name]["iou_thr"],
                                                                        wbf_cfg[name]["conf_type"], wbf_cfg[name]["skip_box_thr"],
                                                                        wbf_cfg[name].get("pre_nms_iou", 0.5))
                thr = best_threshold(tune_c, NUM_CLASSES)
            else:
                thr = 0.25
            res["operating_point"] = operating_point(recs, thr, NUM_CLASSES)
            if args.bootstrap:
                for metric in ("mAP50", "mAP50-95"):
                    res.setdefault("ci95", {})[metric] = bootstrap_ci(
                        recs, lambda rs, m=metric: detection_metrics(rs, CLASS_NAMES)[m], n_boot=args.bootstrap, seed=args.seed)
        results[name] = res
        d = res["detection"]
        log(f"  {name:<28} mAP50 {d['mAP50']:.4f}  mAP50-95 {d['mAP50-95']:.4f}  P {d['precision']:.3f}  R {d['recall']:.3f}")
        configs[name] = recs

    if args.bootstrap and not args.legacy_protocol:  # paired differences vs the best single model
        best_single = max(names, key=lambda n: results[n]["detection"]["mAP50-95"])
        for name in configs:
            if name != best_single:
                results[name]["paired_vs_" + best_single] = bootstrap_ci(
                    configs[name], lambda rs: detection_metrics(rs, CLASS_NAMES)["mAP50-95"], n_boot=args.bootstrap,
                    seed=args.seed, paired_with=configs[best_single])

    lat = latency(names, test, device, args.latency_images, {k: (ens_members[k], wbf_cfg[k]) for k in ens_members}, log) \
        if args.latency_images and not args.legacy_protocol else {}
    out = {"warnings": warnings, "device": device, "n_test": len(test), "n_tune": len(tune), "wbf_config": wbf_cfg,
           "results": results, "latency": lat, "legacy_protocol": args.legacy_protocol}
    (run_dir / "results.json").write_text(json.dumps(out, indent=2, default=float))
    write_tables(run_dir, out)
    log(f"Done. Results in {run_dir}")


def write_tables(run_dir: Path, out: dict):
    import csv

    rows = []
    for name, r in out["results"].items():
        d, c, op = r["detection"], r.get("coco", {}), r.get("operating_point", {})
        ci = r.get("ci95", {})
        rows.append({"model": name, "mAP50": d["mAP50"], "mAP50-95": d["mAP50-95"],
                     "mAP50-95_ci_low": ci.get("mAP50-95", {}).get("low"), "mAP50-95_ci_high": ci.get("mAP50-95", {}).get("high"),
                     "P@maxF1": d["precision"], "R@maxF1": d["recall"], "F1@maxF1": d["f1"],
                     "coco_AP": c.get("AP"), "coco_AP50": c.get("AP50"), "AP_small": c.get("AP_small"),
                     "AP_medium": c.get("AP_medium"), "AP_large": c.get("AP_large"),
                     "op_thr": op.get("conf_thr"), "op_precision": op.get("precision"), "op_recall": op.get("recall"),
                     "op_f1": op.get("f1"), "fp_per_image": op.get("fp_per_image"),
                     "latency_ms": out["latency"].get(name, {}).get("ms_median")})
    with open(run_dir / "table_main.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(run_dir / "table_per_class_ap50_95.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["class"] + list(out["results"]))
        for c in CLASS_NAMES:
            w.writerow([c] + [out["results"][m]["detection"]["per_class"].get(c, {}).get("ap50_95") for m in out["results"]])
    f = lambda v: "--" if v is None else f"{v:.3f}"
    tex = [r"\begin{tabular}{lcccccc}", r"\toprule",
           r"Model & mAP$_{50}$ & mAP$_{50:95}$ (95\% CI) & AP$_{S}$ & AP$_{M}$ & AP$_{L}$ & ms/img \\", r"\midrule"]
    for r in rows:
        ci = f"{f(r['mAP50-95'])} [{f(r['mAP50-95_ci_low'])}, {f(r['mAP50-95_ci_high'])}]" if r["mAP50-95_ci_low"] is not None else f(r["mAP50-95"])
        lat = "--" if r["latency_ms"] is None else f"{r['latency_ms']:.1f}"
        tex.append(f"{r['model']} & {f(r['mAP50'])} & {ci} & {f(r['AP_small'])} & {f(r['AP_medium'])} & {f(r['AP_large'])} & {lat} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (run_dir / "table_main.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()

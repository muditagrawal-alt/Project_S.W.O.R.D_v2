"""
Phase 1 on an EXTERNAL test set: S.W.O.R.D. detectors on real CCTV mock-attack frames (never seen in training).

The S.W.O.R.D. training data are mostly web/stock images; this measures how the detectors transfer to surveillance
footage (1920x1080, small weapons, 2 fps). Built by prepare_mock_attack.py; classes differ from S.W.O.R.D.'s, so:
    agnostic  every weapon is one class ("is there a weapon here?") — the primary result
    grouped   handgun (Pistol) / long_gun (Shotgun, Assault Rifle, Sniper Rifle, SMG, Machine Gun) / knife (Knife);
              predictions of Grenade, Sword and Rocket Launcher go to an "other" group without ground truth
Predictions at conf 0.001 (AP from all predictions); operating points at 0.25 and 0.5 (the pipeline's alarm
threshold); false alarms on weapon-free frames; segment-level (cluster) bootstrap CIs, because frames inside a
segment are strongly correlated. Inference sizes 640 (training size), 1280 and 1920 (native scale for 1080p):
the weapons here are ~37x55 px at 1080p, i.e. ~12x18 px at 640.
WBF uses the documented defaults (DEFAULT_WBF, equal weights). Pre-declared camera split for any configuration
choice made from these results: DEV = Cam1 + Cam5 (select), TEST = Cam7 (report; different viewpoints).

    python research/tools/integration/eval_phase1_external.py --out research/runs/<id>_phase1-external-mockattack
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.benchmark import DEFAULT_WBF, MODELS, file_digest, fuse, load_model  # noqa: E402
from ensemble.classes import CLASS_NAMES  # noqa: E402
from ensemble.metrics import coco_metrics, detection_metrics, operating_point, precompute_tp  # noqa: E402

GROUPS = ["handgun", "long_gun", "knife", "other"]
SPLITS = {"dev_cam1_cam5": ("Cam1", "Cam5"), "test_cam7": ("Cam7",)}
SWORD_TO_GROUP = {"Pistol": 0, "Shotgun": 1, "Assault Rifle": 1, "Sniper Rifle": 1, "SMG": 1, "Machine Gun": 1,
                  "Knife": 2, "Grenade": 3, "Sword": 3, "Rocket Launcher": 3}
LABEL_TO_GROUP = {"Handgun": 0, "Short_rifle": 1, "Knife": 2}
PRED_GROUP = np.array([SWORD_TO_GROUP[n] for n in CLASS_NAMES])


def load_sequences(seq_dir: Path):
    recs: list = []
    segs: list = []
    for f in sorted(seq_dir.glob("*.json")):
        if f.name in ("index.json", "manifest.json"):
            continue
        doc = json.loads(f.read_text())
        for fr in doc["frames"]:
            b = np.array([x["bbox"] for x in fr["boxes"]], dtype=np.float64).reshape(-1, 4)
            recs.append({"image_id": len(recs), "path": fr["image"], "width": doc["width"], "height": doc["height"],
                         "segment": doc["segment"], "camera": doc["camera"], "gt_boxes": b,
                         "gt_group": np.array([LABEL_TO_GROUP[x["label"]] for x in fr["boxes"]], dtype=np.int64)})
        segs.append(doc["segment"])
    return recs, segs


def predict(name: str, imgsz: int, recs: list, device: str, cache_dir: Path, log) -> list:
    kind, weights = MODELS[name]
    weights = Path(weights)
    cache = cache_dir / f"{name}_{imgsz}_{file_digest(weights)}_{len(recs)}.pkl"
    if cache.exists():
        log(f"  [cache] {name}@{imgsz}")
        return pickle.loads(cache.read_bytes())
    model = load_model(kind, weights)
    out, t0 = [], time.time()
    paths = [r["path"] for r in recs]
    for s in range(0, len(paths), 16):
        for r, res in zip(recs[s:s + 16], model.predict(paths[s:s + 16], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                                                         device=device, verbose=False)):
            b = res.boxes
            out.append({"pred_cls": b.cls.cpu().numpy().astype(np.int64), "pred_scores": b.conf.cpu().numpy().astype(np.float64),
                        "pred_boxes": b.xyxy.cpu().numpy().astype(np.float64)})
    sec = time.time() - t0
    log(f"  {name}@{imgsz}: {len(paths)} frames in {sec:.0f}s ({1000 * sec / len(paths):.1f} ms/frame incl. decode)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def as_records(recs: list, preds: list, mode: str) -> list:
    out = []
    for r, p in zip(recs, preds):
        if mode == "agnostic":
            gt_cls, pc = np.zeros(len(r["gt_group"]), np.int64), np.zeros(len(p["pred_cls"]), np.int64)
        else:
            gt_cls, pc = r["gt_group"], PRED_GROUP[p["pred_cls"]] if len(p["pred_cls"]) else p["pred_cls"]
        out.append({"image_id": r["image_id"], "path": r["path"], "width": r["width"], "height": r["height"],
                    "segment": r["segment"], "camera": r["camera"], "gt_cls": gt_cls, "gt_boxes": r["gt_boxes"],
                    "pred_cls": pc, "pred_scores": p["pred_scores"], "pred_boxes": p["pred_boxes"]})
    return out


def frame_alarm_rates(records: list, thr: float) -> dict:
    """Frame-level view (what an alarm sees): weapon-free frames with any detection >= thr, weapon frames with one."""
    neg = [r for r in records if len(r["gt_cls"]) == 0]
    pos = [r for r in records if len(r["gt_cls"]) > 0]
    fa = sum(bool((r["pred_scores"] >= thr).any()) for r in neg)
    hit = sum(bool((r["pred_scores"] >= thr).any()) for r in pos)
    return {"thr": thr, "weapon_free_frames": len(neg), "weapon_free_frames_with_detection": fa,
            "frame_false_alarm_rate": round(fa / max(len(neg), 1), 4), "weapon_frames": len(pos),
            "weapon_frames_with_detection": hit, "frame_hit_rate": round(hit / max(len(pos), 1), 4)}


def cluster_bootstrap(records: list, metric, n_boot: int = 300, seed: int = 0) -> dict:
    by_seg = defaultdict(list)
    for r in records:
        by_seg[r["segment"]].append(r)
    keys = sorted(by_seg)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        vals.append(metric([r for i in pick for r in by_seg[keys[i]]]))
    return {"point": float(metric(records)), "low": float(np.quantile(vals, 0.025)), "high": float(np.quantile(vals, 0.975)),
            "n_boot": n_boot, "unit": "segment", "n_clusters": len(keys)}


def evaluate(recs: list, preds: list, n_boot: int) -> dict:
    ag = as_records(recs, preds, "agnostic")
    gr = as_records(recs, preds, "grouped")
    precompute_tp(ag)
    precompute_tp(gr)
    det = detection_metrics(ag, ["weapon"])
    out = {"agnostic": {"mAP50": det["mAP50"], "mAP50-95": det["mAP50-95"], "precision_maxF1": det["precision"],
                        "recall_maxF1": det["recall"], "f1_max": det["f1"], "n_gt": det["n_gt"], "n_pred": det["n_pred"],
                        "coco": coco_metrics(ag, ["weapon"]),
                        "operating_points": {str(t): operating_point(ag, t, 1) for t in (0.25, 0.5)},
                        "frame_level": {str(t): frame_alarm_rates(ag, t) for t in (0.25, 0.5)},
                        "ap50_ci": cluster_bootstrap(ag, lambda rs: detection_metrics(rs, ["weapon"])["mAP50"], n_boot)},
           "grouped": detection_metrics(gr, GROUPS),
           "per_camera_agnostic_ap50": {}}
    for cam in sorted({r["camera"] for r in ag}):
        sub = [r for r in ag if r["camera"] == cam]
        out["per_camera_agnostic_ap50"][cam] = round(detection_metrics(sub, ["weapon"])["mAP50"], 4)
    out["splits_agnostic"] = {}
    for split, cams in SPLITS.items():
        sub = [r for r in ag if r["camera"] in cams]
        d = detection_metrics(sub, ["weapon"])
        out["splits_agnostic"][split] = {"frames": len(sub), "mAP50": d["mAP50"], "mAP50-95": d["mAP50-95"],
                                         "operating_point_0.5": operating_point(sub, 0.5, 1),
                                         "frame_level_0.5": frame_alarm_rates(sub, 0.5)}
    # which S.W.O.R.D. classes fire on weapon-free frames (score >= 0.5): the false-alarm vocabulary
    fa: dict = defaultdict(int)
    for r, p in zip(recs, preds):
        if len(r["gt_group"]) == 0:
            for c in p["pred_cls"][p["pred_scores"] >= 0.5]:
                fa[CLASS_NAMES[int(c)]] += 1
    out["false_alarm_classes_at_0.5"] = dict(sorted(fa.items(), key=lambda kv: -kv[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", default=str(PROJECT_ROOT / "research/data/mock_attack/sequences"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=KIND:PATH",
                    help="evaluate these weights instead of the default three (KIND = yolo | rtdetr), repeatable")
    ap.add_argument("--no-wbf", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], help="model@imgsz configurations to leave out (e.g. RT-DETR-L@1920); "
                                                           "WBF at a size needs all three models")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(out_dir / "log.txt", "a")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    if args.model:
        MODELS.clear()
        for spec in args.model:
            name, rest = spec.split("=", 1)
            kind, path = rest.split(":", 1)
            MODELS[name] = (kind, Path(path).resolve())
    recs, segs = load_sequences(Path(args.sequences))
    log(f"{len(recs)} frames, {len(segs)} segments, {sum(len(r['gt_group']) for r in recs)} weapons, "
        f"{sum(len(r['gt_group']) == 0 for r in recs)} weapon-free frames")
    results = {}
    for imgsz in args.imgsz:
        per_model = {}
        for name in MODELS:
            if f"{name}@{imgsz}" in args.skip:
                log(f"  [skipped by request] {name}@{imgsz}")
                continue
            per_model[name] = predict(name, imgsz, recs, args.device, out_dir / "cache", log)
            results[f"{name}@{imgsz}"] = evaluate(recs, per_model[name], args.n_boot)
            log(f"  {name}@{imgsz}: agnostic AP50 {results[f'{name}@{imgsz}']['agnostic']['mAP50']:.3f}")
        if args.no_wbf or len(MODELS) < 2 or len(per_model) < len(MODELS):
            log(f"  [skipped] WBF@{imgsz} (needs all three models)")
            (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
            continue
        per_model_recs = [[{**r, **p} for r, p in zip(recs, per_model[n])] for n in MODELS]
        fused = fuse(per_model_recs,[1.0] * len(MODELS), DEFAULT_WBF["iou_thr"], DEFAULT_WBF["conf_type"],
                     DEFAULT_WBF["skip_box_thr"], DEFAULT_WBF["pre_nms_iou"])
        wbf_preds = [{"pred_cls": f["pred_cls"], "pred_scores": f["pred_scores"], "pred_boxes": f["pred_boxes"]} for f in fused]
        results[f"WBF@{imgsz}"] = evaluate(recs, wbf_preds, args.n_boot)
        log(f"  WBF@{imgsz}: agnostic AP50 {results[f'WBF@{imgsz}']['agnostic']['mAP50']:.3f}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))

    rows = ["| Config | AP50 all (95% CI, segment bootstrap) | AP50 DEV Cam1+5 | AP50 TEST Cam7 | AP50-95 | AP_small | P@0.5 | R@0.5 | FP/img @0.5 | weapon-free frames alarmed @0.5 | weapon frames hit @0.5 | grouped mAP50 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        a = v["agnostic"]
        ci, op, fl = a["ap50_ci"], a["operating_points"]["0.5"], a["frame_level"]["0.5"]
        small = a["coco"].get("AP_small")
        sp = v["splits_agnostic"]
        rows.append(f"| {k} | {a['mAP50']:.3f} ({ci['low']:.3f}–{ci['high']:.3f}) | {sp['dev_cam1_cam5']['mAP50']:.3f} | "
                    f"{sp['test_cam7']['mAP50']:.3f} | {a['mAP50-95']:.3f} | "
                    f"{'n/a' if small is None else f'{small:.3f}'} | {op['micro_precision']:.3f} | {op['micro_recall']:.3f} | "
                    f"{op['fp_per_image']:.3f} | {100 * fl['frame_false_alarm_rate']:.1f}% | {100 * fl['frame_hit_rate']:.1f}% | "
                    f"{v['grouped']['mAP50']:.3f} |")
    table = "\n".join(rows)
    (out_dir / "results_table.md").write_text(table + "\n")
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "args": vars(args), "frames": len(recs), "segments": segs,
                "weights": {n: {"path": str(Path(w[1]).resolve().relative_to(PROJECT_ROOT)), "sha256_12": file_digest(Path(w[1]))}
                            for n, w in MODELS.items()},
                "wbf": DEFAULT_WBF, "platform": platform.platform(), "python": platform.python_version()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log("\n" + table)


if __name__ == "__main__":
    main()

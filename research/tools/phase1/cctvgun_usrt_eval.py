"""
Compare with the CCTV-Gun benchmark (Yellapragada et al., arXiv:2303.10703): handgun AP at IoU 0.5 on its USRT
image set (3,294 images, its own handgun boxes), the "cross-dataset" protocol in which the detector never saw USRT.
Published (Table 5, train UCF+MGD -> test all of USRT): Faster R-CNN 22.1, Deformable DETR 15.9, DetectoRS 25.7,
Swin-T Faster R-CNN 26.2, ConvNeXt-T Faster R-CNN 27.3.

Detections come from recorded replays (the system's detector outputs on every mock-attack frame, which is USRT),
optionally with a profile's detection filters applied. Two readings:
  handgun  only detections of the Pistol class count
  any      any weapon class counts as a handgun detection (the system raises alarms on any weapon class)
AP uses COCO-style 101-point interpolation (faster-coco-eval), as the benchmark's MMDetection evaluation does.
CIs: bootstrap over the 20 recorded segments. Detections were recorded at score >= 0.1 (the system's operating
range), which slightly understates AP compared with evaluating every low-score box.

    python research/tools/phase1/cctvgun_usrt_eval.py --replay <run> [--profile cctv-zoom] --label <name> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.classes import CLASS_NAMES  # noqa: E402
from ensemble.metrics import coco_metrics, detection_metrics, precompute_tp  # noqa: E402

ANN = PROJECT_ROOT / "research/data/cctv_gun/data/usrt/annotation_detection/annotations_all.json"
SEQ = PROJECT_ROOT / "research/data/mock_attack/sequences"
PISTOL = CLASS_NAMES.index("Pistol")
PUBLISHED = {"Faster R-CNN+FPN (R50)": 22.1, "Deformable DETR (R50)": 15.9, "DetectoRS (R50)": 25.7,
             "Swin-T Faster R-CNN+FPN": 26.2, "ConvNeXt-T Faster R-CNN+FPN": 27.3}


def filters_of(profile: str | None) -> dict:
    if not profile:
        return {"zoom_min_score": 0.0, "max_area_frac": 1.0, "zoom_min_window": 0, "zoom_modes": (),
                "verify_mode": "none", "verify_min": 0.0}
    from pipeline.profiles import list_profiles

    p = list_profiles()[profile]
    return {"zoom_min_score": float(p.get("zoom_min_score", 0.0)), "max_area_frac": float(p.get("weapon_max_area", 1.0)),
            "zoom_min_window": int(p.get("zoom_min_window", 0)), "zoom_modes": tuple(p.get("weapon_zoom", ())),
            "verify_mode": p.get("verify_mode", "none") if p.get("weapon_verifier") else "none",
            "verify_min": float(p.get("verify_min", 0.0)) if p.get("weapon_verifier") else 0.0}


def scored(det: dict, f: dict):
    """Detection score after the profile's verifier fusion (None if the verifier gates it out)."""
    if f.get("verify_mode", "none") == "none" and not f.get("verify_min"):
        return det["score"]
    if "verify" not in det:
        raise ValueError("profile uses the verifier but the recording has no 'verify' scores (apply_weapon_verifier.py)")
    if det["verify"] < f["verify_min"]:
        return None
    from tracking.verifier import fuse_scores

    return float(fuse_scores([det.get("det_score", det["score"])], [det["verify"]], f["verify_mode"])[0])


def keep(det: dict, persons: list, shape, f: dict) -> bool:
    b = det["bbox"]
    if det.get("src") == "zoom" and det["score"] < f["zoom_min_score"]:
        return False
    if (b[2] - b[0]) * (b[3] - b[1]) > f["max_area_frac"] * shape[0] * shape[1]:
        return False
    if f["zoom_min_window"] and det.get("src") == "zoom":
        from tracking.zoom import ZoomConfig, from_allowed_window

        return from_allowed_window(b, persons, shape, ZoomConfig(modes=f["zoom_modes"] or ("hands",), min_window=f["zoom_min_window"]))
    return True


def records(replay: Path, f: dict, mode: str, score_key: str = "score") -> list:
    ann = json.loads(ANN.read_text())
    gt = defaultdict(list)
    for a in ann["annotations"]:
        if a["category_id"] == 2:
            x, y, w, h = a["bbox"]
            gt[a["image_id"]].append([x, y, x + w, y + h])
    frame_of = {}   # image file name -> (segment, index)
    for sj in SEQ.glob("Cam*.json"):
        for i, fr in enumerate(json.loads(sj.read_text()).get("frames", [])):
            frame_of[Path(fr["image"]).name] = (sj.name, i)
    docs = {fj.parent.name: json.loads(fj.read_text()) for fj in replay.glob("segments/Cam*/frames.json")}
    by_video = {d["video"]: d for d in docs.values()}
    out = []
    for n, img in enumerate(ann["images"]):
        seg, i = frame_of[img["file_name"]]
        d = by_video[seg]
        fr = d["frames"][i]
        dets = [{**x, "score": scored(x, f)} for x in fr["weapon_detections"]
                if keep(x, fr["persons"], (d["height"], d["width"]), f) and (mode == "any" or x["cls"] == PISTOL)]
        dets = [x for x in dets if x["score"] is not None]
        g = np.asarray(gt.get(img["id"], []), float).reshape(-1, 4)
        out.append({"image_id": n, "path": img["file_name"], "width": img["width"], "height": img["height"],
                    "segment": seg, "camera": seg.split("-")[0],
                    "gt_cls": np.zeros(len(g), np.int64), "gt_boxes": g,
                    "pred_cls": np.zeros(len(dets), np.int64),
                    "pred_scores": np.asarray([x.get(score_key, x["score"]) for x in dets], float),
                    "pred_boxes": np.asarray([x["bbox"] for x in dets], float).reshape(-1, 4)})
    return out


def ap50(recs: list) -> float:
    return 100 * (coco_metrics(recs, ["handgun"]).get("AP50") or 0.0)   # no detections at all -> AP 0


def cluster_ci(recs: list, n_boot: int = 300, seed: int = 0) -> list:
    by = defaultdict(list)
    for r in recs:
        by[r["segment"]].append(r)
    keys, rng, vals = sorted(by), np.random.default_rng(seed), []
    for _ in range(n_boot):
        vals.append(ap50([r for k in rng.integers(0, len(keys), len(keys)) for r in by[keys[k]]]))
    return [round(float(np.quantile(vals, 0.025)), 1), round(float(np.quantile(vals, 0.975)), 1)]


def evaluate(replay: Path, profile: str | None, score_key: str = "score", n_boot: int = 300) -> dict:
    f = filters_of(profile)
    res = {}
    for mode in ("handgun", "any"):
        recs = records(replay, f, mode, score_key)
        precompute_tp(recs)
        cam7 = [r for r in recs if r["camera"] == "Cam7"]
        res[mode] = {"AP50_all_usrt": round(ap50(recs), 1), "AP50_all_usrt_ci95": cluster_ci(recs, n_boot),
                     "AP50_cam7": round(ap50(cam7), 1), "AP50_all_usrt_voc_style": round(100 * detection_metrics(recs, ["handgun"])["mAP50"], 1),
                     "images": len(recs), "gt_handguns": int(sum(len(r["gt_boxes"]) for r in recs)),
                     "detections": int(sum(len(r["pred_scores"]) for r in recs))}
    return {"filters": {k: list(v) if isinstance(v, tuple) else v for k, v in f.items()}, "results": res}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", action="append", required=True, help="label=run_dir[:profile]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"created": datetime.now().isoformat(timespec="seconds"), "annotations": str(ANN.relative_to(PROJECT_ROOT)),
              "published_cross_dataset_handgun_ap50": PUBLISHED, "systems": {}}
    for spec in args.replay:
        label, rest = spec.split("=", 1)
        run, _, profile = rest.partition(":")
        result["systems"][label] = {"replay": run, "profile": profile or None, **evaluate(Path(run), profile or None, n_boot=args.n_boot)}
        print(label, json.dumps(result["systems"][label]["results"]), flush=True)
    (out / "results.json").write_text(json.dumps(result, indent=2))
    lines = ["| system | handgun AP50, all USRT (95% CI) | any-weapon AP50, all USRT (95% CI) | handgun AP50, Cam7 |", "|---|---|---|---|"]
    for k, v in PUBLISHED.items():
        lines.append(f"| {k} (published, train UCF+MGD) | {v} | — | — |")
    for label, s in result["systems"].items():
        h, a = s["results"]["handgun"], s["results"]["any"]
        lines.append(f"| {label} | {h['AP50_all_usrt']} ({h['AP50_all_usrt_ci95'][0]}–{h['AP50_all_usrt_ci95'][1]}) | "
                     f"{a['AP50_all_usrt']} ({a['AP50_all_usrt_ci95'][0]}–{a['AP50_all_usrt_ci95'][1]}) | {h['AP50_cam7']} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

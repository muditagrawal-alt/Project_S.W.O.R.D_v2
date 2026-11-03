"""
Person-zoom weapon detection on the external mock-attack CCTV set (Phase 1 protocol), using the tracked persons and
wrist keypoints a full-pipeline run recorded for every frame.

Variants: full frame only (cached predictions), person crops, hand crops, and their unions with the full frame
(class-wise NMS). Pre-declared selection: best class-agnostic AP50 on DEV (Cam1+Cam5); TEST (Cam7) is reported.

    python research/tools/integration/person_zoom_eval.py --weights <best.pt> --full-cache <cache .pkl @1280> \\
        --recording research/runs/20261024-231132_full-pipeline-default-yolo26s-640 --out research/runs/<id>_person-zoom
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_phase1_external import evaluate, load_sequences  # noqa: E402
from tracking.zoom import ZoomConfig, detect_in_windows, merge_detections, zoom_windows  # noqa: E402

SEQ_DIR = PROJECT_ROOT / "research/data/mock_attack/sequences"


def persons_by_frame(recording: Path) -> dict:
    out = {}
    for fj in sorted(recording.glob("segments/Cam*/frames.json")):
        doc = json.loads(fj.read_text())
        for f in doc["frames"]:
            out[(fj.parent.name, f["frame_idx"])] = f["persons"]
    return out


def crop_predictions(model, recs, persons, mode: str, imgsz: int, device: str, cfg: ZoomConfig, log) -> list:
    out, t0, n_win = [], time.time(), 0
    for i, r in enumerate(recs):
        img = cv2.imread(r["path"])
        win = zoom_windows(persons.get((r["segment"], r["frame_idx"]), []), img.shape,
                           ZoomConfig(**{**cfg.__dict__, "modes": (mode,)}))
        n_win += len(win)
        b, s, c = detect_in_windows(model, img, win, imgsz=imgsz, conf=0.001, device=device, drop_border=cfg.drop_border)
        out.append({"pred_boxes": b, "pred_scores": s, "pred_cls": c})
        if (i + 1) % 1000 == 0:
            log(f"  {mode}: {i + 1}/{len(recs)} frames, {n_win} crops, {time.time() - t0:.0f}s")
    log(f"  {mode}: {n_win} crops in {time.time() - t0:.0f}s")
    return out


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--full-cache", required=True, help="eval_phase1_external.py cache (.pkl) of full-frame predictions")
    ap.add_argument("--recording", required=True, help="full-pipeline run with segments/Cam*/frames.json (persons, keypoints)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-imgsz", type=int, default=640)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fp = open(out / "log.txt", "a")

    def log(m):
        line = f"[{datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    recs, _ = load_sequences(SEQ_DIR)
    # frame_idx per record (load_sequences keeps file order; recover the index within each segment)
    seen = {}
    for r in recs:
        r["frame_idx"] = seen.get(r["segment"], 0)
        seen[r["segment"]] = r["frame_idx"] + 1
    persons = persons_by_frame(Path(args.recording))
    missing = sum((r["segment"], r["frame_idx"]) not in persons for r in recs)
    log(f"{len(recs)} frames; persons recorded for {len(recs) - missing}")
    full = pickle.loads(Path(args.full_cache).read_bytes())
    assert len(full) == len(recs)
    model = YOLO(args.weights)
    cfg = ZoomConfig()
    crops = {}
    for mode in ("person", "hands"):
        cache = out / f"crops_{mode}_{args.crop_imgsz}.pkl"
        if cache.exists():
            crops[mode] = pickle.loads(cache.read_bytes())
        else:
            crops[mode] = crop_predictions(model, recs, persons, mode, args.crop_imgsz, args.device, cfg, log)
            cache.write_bytes(pickle.dumps(crops[mode]))
    sources = {"full": full, **crops}
    results = {}
    for k in range(1, 4):
        for combo in itertools.combinations(["full", "person", "hands"], k):
            name = "+".join(combo)
            preds = []
            for i in range(len(recs)):
                parts = [(sources[s][i]["pred_boxes"], sources[s][i]["pred_scores"], sources[s][i]["pred_cls"]) for s in combo]
                b, s, c = merge_detections(parts) if len(parts) > 1 else parts[0]
                order = np.argsort(-np.asarray(s))[:300]
                preds.append({"pred_boxes": np.asarray(b)[order], "pred_scores": np.asarray(s)[order], "pred_cls": np.asarray(c)[order]})
            results[name] = evaluate(recs, preds, args.n_boot)
            a = results[name]["agnostic"]
            sp = results[name]["splits_agnostic"]
            log(f"{name:20s} AP50 all {a['mAP50']:.3f} [{a['ap50_ci']['low']:.3f}, {a['ap50_ci']['high']:.3f}] | DEV {sp['dev_cam1_cam5']['mAP50']:.3f} "
                f"| TEST {sp['test_cam7']['mAP50']:.3f} | P@.5 {a['operating_points']['0.5']['micro_precision']:.3f} "
                f"R@.5 {a['operating_points']['0.5']['micro_recall']:.3f} | handgun {results[name]['grouped']['per_class'].get('handgun', {}).get('ap50', 0):.3f}")
    best = max(results, key=lambda k: results[k]["splits_agnostic"]["dev_cam1_cam5"]["mAP50"])
    log(f"DEV-selected variant: {best}")
    (out / "results.json").write_text(json.dumps({"selected_on_dev": best, "results": results}, indent=2, default=float))
    rows = ["| Variant | AP50 all (95% CI) | DEV Cam1+5 | TEST Cam7 | AP_small | P@0.5 | R@0.5 | weapon-free frames alarmed @0.5 | handgun / long gun AP50 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, v in results.items():
        a, sp, g = v["agnostic"], v["splits_agnostic"], v["grouped"]["per_class"]
        small = a["coco"].get("AP_small")
        rows.append(f"| {k}{' (dev-selected)' if k == best else ''} | {a['mAP50']:.3f} ({a['ap50_ci']['low']:.3f}–{a['ap50_ci']['high']:.3f}) | "
                    f"{sp['dev_cam1_cam5']['mAP50']:.3f} | {sp['test_cam7']['mAP50']:.3f} | {'n/a' if small is None else f'{small:.3f}'} | "
                    f"{a['operating_points']['0.5']['micro_precision']:.3f} | {a['operating_points']['0.5']['micro_recall']:.3f} | "
                    f"{100 * a['frame_level']['0.5']['frame_false_alarm_rate']:.1f}% | "
                    f"{g.get('handgun', {}).get('ap50', 0):.3f} / {g.get('long_gun', {}).get('ap50', 0):.3f} |")
    (out / "results_table.md").write_text("\n".join(rows) + "\n")
    (out / "manifest.json").write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                                                   "zoom_config": cfg.__dict__, "selection_rule": "best class-agnostic AP50 on DEV Cam1+Cam5"},
                                                  indent=2, default=str))
    log("\n" + "\n".join(rows))


if __name__ == "__main__":
    main()

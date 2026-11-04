"""
Swap the weapon detector in recorded pipeline runs without re-running Phases 2-4: re-detect weapons on exactly the
recorded frames, replace `weapon_detections` (weapons are re-derived from them by the evaluation code), keep the
recorded persons, poses and actions. Writes a parallel recording that every evaluation tool accepts.

    python research/tools/integration/detector_replay.py --recording research/runs/<pipeline run> \\
        --weights research/runs/<detector run>/train/weights/best.pt --imgsz 640 --out research/runs/<id>_replay-<name>

Sources: Mendeley / UCF-Crime videos (frame stride inferred from the recorded and native fps) and mock-attack
image sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
DATA = PROJECT_ROOT / "research" / "data"


def source_frames(doc: dict):
    """Yield (frame_idx, image) for the recorded frames of one recording."""
    vid = doc["video"]
    if vid.endswith(".json"):   # mock-attack image sequence
        seq = json.loads((DATA / "mock_attack" / "sequences" / vid).read_text())
        for f in seq["frames"]:
            yield f["frame_idx"], cv2.imread(f["image"])
        return
    parts = vid.split("/")
    if vid == "vtest.avi":
        path = DATA / "videos" / "vtest.avi"
    elif parts[0] == "mendeley":
        path = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset" / Path(*parts[1:])
    elif parts[0] == "ucf":
        path = DATA / "ucf_crime" / "extracted" / "UCF_Crimes" / "Videos" / Path(*parts[1:])
    else:
        raise ValueError(vid)
    cap = cv2.VideoCapture(str(path))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native / doc["fps"])))
    src, idx = -1, 0
    while True:
        ok, img = cap.read()
        src += 1
        if not ok:
            break
        if src % stride:
            continue
        yield idx, img
        idx += 1
    cap.release()


def main():
    from ultralytics import RTDETR, YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.1, help="same detector threshold as the live pipeline")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--zoom", default="", help="person-zoom crops merged with the full frame: '', person, hands, person+hands")
    ap.add_argument("--zoom-imgsz", type=int, default=640)
    ap.add_argument("--zoom-conf", type=float, default=None, help="detector threshold on the zoom crops (default: --conf; the live pipeline uses max(conf, zoom_min_score))")
    ap.add_argument("--zoom-min-window", type=int, default=0, help="skip zoom windows smaller than this (tracking/zoom.py)")
    ap.add_argument("--tta-scales", nargs="*", type=int, default=[], help="test-time augmentation: extra full-frame "
                                                                         "image sizes merged with --imgsz")
    ap.add_argument("--tta-flip", action="store_true", help="test-time augmentation: also the horizontally flipped frame")
    ap.add_argument("--zoom-flip", action="store_true", help="test-time augmentation: also the horizontally flipped zoom crops")
    ap.add_argument("--reverse", action="store_true", help="process recordings in reverse order (a second process can "
                                                           "share the work; finished recordings are skipped)")
    ap.add_argument("--skip", nargs="*", default=[], help="skip recordings whose folder name contains any of these")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec, out = Path(args.recording), Path(args.out)
    model = (RTDETR if "rtdetr" in Path(args.weights).name.lower() else YOLO)(args.weights)
    from tracking.zoom import ZoomConfig, detect_full_frame, detect_in_windows, merge_detections, zoom_windows

    zoom_modes = tuple(m for m in args.zoom.split("+") if m)
    files = sorted(list(rec.glob("videos/*/frames.json")) + list(rec.glob("segments/*/frames.json")), reverse=args.reverse)
    t0, n_frames = time.time(), 0
    files = [f for f in files if not any(k in f.parent.name for k in args.skip)]
    for fj in files:
        dst = out / fj.parent.relative_to(rec) / "frames.json"
        if dst.exists():
            continue
        doc = json.loads(fj.read_text())
        by_idx = {f["frame_idx"]: f for f in doc["frames"]}
        for idx, img in source_frames(doc):
            f = by_idx.get(idx)
            if f is None or img is None:
                continue
            b, s, c = detect_full_frame(model, img, args.imgsz, args.tta_scales, args.tta_flip, args.conf, args.device)
            if zoom_modes:   # same code path as the live pipeline (tracking/zoom.py)
                win = zoom_windows(f["persons"], img.shape, ZoomConfig(modes=zoom_modes, min_window=args.zoom_min_window))
                zb, zs, zc = detect_in_windows(model, img, win, imgsz=args.zoom_imgsz, device=args.device,
                                               conf=args.conf if args.zoom_conf is None else args.zoom_conf,
                                               flip=args.zoom_flip)
                b, s, c, origin = merge_detections([(b, s, c), (zb, zs, zc)], return_source=True)
                src = ["full" if o == 0 else "zoom" for o in origin]
            else:
                src = None
            f["weapon_detections"] = [{"bbox": [round(float(v), 1) for v in bb], "cls": int(cc), "score": round(float(ss), 3),
                                       **({"src": src[k]} if src is not None else {})}
                                      for k, (bb, ss, cc) in enumerate(zip(b, s, c))]
            f["weapons"] = []   # re-derived from weapon_detections by the evaluation code (tracking.pipeline.rederive_weapons)
            n_frames += 1
        tta = "".join(f"+{sz}" for sz in args.tta_scales) + (" +flip" if args.tta_flip else "")
        doc["sources"] = {**doc.get("sources", {}), "weapon_detector": f"{Path(args.weights).name} @ imgsz {args.imgsz}{tta}"
                          + (f" + zoom {args.zoom} @ {args.zoom_imgsz}" if zoom_modes else "")
                          + (" +flip" if zoom_modes and args.zoom_flip else "") + " (replayed)"}
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(doc))
        # let run_full_pipeline.py --reevaluate score the replay like any recording (timings are the original run's)
        summ = fj.parent / "summary.json"
        if summ.exists():
            sd = json.loads(summ.read_text())
            sd["note"] = "copied from the original recording; timings are not those of the replayed detector"
            (dst.parent / "summary.json").write_text(json.dumps(sd, indent=2))
            (dst.parent / "evaluation.json").write_text("{}")   # placeholder, recomputed by --reevaluate
    (out / "manifest.json").write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                                                   "recordings": len(files), "frames_redetected": n_frames,
                                                   "minutes": round((time.time() - t0) / 60, 1)}, indent=2))
    print(f"{len(files)} recordings, {n_frames} frames re-detected in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()

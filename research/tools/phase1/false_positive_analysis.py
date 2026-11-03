"""
Phase 1 error analysis: where do the weapon detector's false positives come from on CCTV?

For a replay recorded with detection origins (detector_replay.py --zoom ...), every raw detection on the mock-attack
sequences is matched to the ground truth (IoU >= 0.1) and broken down by
  - origin: full frame vs zoom crop; for zoom detections, the natural side of the hand window it came from
  - box area as a share of the frame, and class
on DEV (Cam1+Cam5), TEST (Cam7) and the weapon-free vtest clip. The same statistics of the TRAINING labels
(research/data/cctv_det: box area share per source; research/data/cctv_det_zoom: zoom-crop sizes) are reported next to
them; detection filters derived from the training side only (pipeline profiles: weapon_max_area, zoom_min_window).

    python research/tools/phase1/false_positive_analysis.py --replay <replay run> --weights <detector .pt> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.zoom import ZoomConfig, zoom_windows  # noqa: E402

SEQ = PROJECT_ROOT / "research/data/mock_attack/sequences"
DATA = PROJECT_ROOT / "research/data"
WINDOW_BINS = ((0, 96), (96, 160), (160, 240), (240, 400), (400, 10_000))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def window_side(box, persons, shape, cfg) -> float:
    """Natural side of the smallest hand window containing the box centre (0 if none)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    sides = [n for w, n in zoom_windows(persons, shape, cfg, with_natural_side=True)
             if w[0] <= cx <= w[2] and w[1] <= cy <= w[3]]
    return min(sides) if sides else 0.0


def pct(a, qs=(10, 50, 90, 99)) -> dict:
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs} if len(a) else {}


def training_stats() -> dict:
    out = {}
    for split in ("train", "val"):
        areas = defaultdict(list)
        for lf in (DATA / "cctv_det" / "labels" / split).glob("*.txt"):
            src = lf.name.split("_")[0]
            for line in lf.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    areas[src].append(100 * float(p[3]) * float(p[4]))
        out[split] = {src: {"boxes": len(a), "area_pct": pct(a, (50, 99, 99.9)), "max_area_pct": round(max(a), 2)}
                      for src, a in areas.items()}
    try:
        from PIL import Image

        sides = [min(Image.open(f).size) for f in (DATA / "cctv_det_zoom" / "images" / "train").glob("zoom*")]
        out["zoom_crops_train"] = {"crops": len(sides), "min_side_px": int(min(sides)), "side_px": pct(sides, (1, 50, 99))}
    except Exception as e:   # noqa: BLE001
        out["zoom_crops_train"] = {"error": str(e)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--weights", required=True, help="detector of the replay (for class names)")
    ap.add_argument("--zoom-modes", default="hands")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    names = YOLO(args.weights).names
    cfg = ZoomConfig(modes=tuple(m for m in args.zoom_modes.split("+") if m))
    rows = defaultdict(list)   # group -> (src, score, tp, window side, area %, class)
    for fj in sorted(Path(args.replay).glob("segments/*/frames.json")):
        d = json.loads(fj.read_text())
        group = "vtest" if d["video"].startswith("vtest") else "dev" if d["video"].startswith(("Cam1", "Cam5")) else "test"
        gt = json.loads((SEQ / d["video"]).read_text())["frames"] if group != "vtest" else None
        shape, frame_area = (d["height"], d["width"]), d["height"] * d["width"]
        for i, f in enumerate(d["frames"]):
            boxes = [b["bbox"] for b in gt[i]["boxes"]] if gt else []
            for w in f["weapon_detections"]:
                b = w["bbox"]
                src = w.get("src", "full")
                rows[group].append((src, w["score"], any(iou(b, g) >= 0.1 for g in boxes),
                                    window_side(b, f["persons"], shape, cfg) if src == "zoom" else -1.0,
                                    100 * (b[2] - b[0]) * (b[3] - b[1]) / frame_area, names[w["cls"]]))
    report = {"created": datetime.now().isoformat(timespec="seconds"), "replay": args.replay, "weights": args.weights,
              "min_score": args.min_score, "training": training_stats(), "groups": {}}
    for group, rs in rows.items():
        g = {}
        for src in ("full", "zoom"):
            r = [x for x in rs if x[0] == src]
            hi = [x for x in r if x[1] >= args.min_score]
            g[src] = {"detections": len(r), "true_pos": sum(x[2] for x in r), f"score>={args.min_score}": len(hi),
                      f"true_pos_score>={args.min_score}": sum(x[2] for x in hi),
                      "fp_area_pct": pct([x[4] for x in hi if not x[2]]), "tp_area_pct": pct([x[4] for x in hi if x[2]]),
                      "fp_classes": dict(Counter(x[5] for x in hi if not x[2]).most_common(5))}
            if src == "zoom":
                g[src]["by_window_side_px"] = {f"{lo}-{hi_}": {"detections": sum(lo <= x[3] < hi_ for x in r),
                                                               "true_pos": sum(x[2] and lo <= x[3] < hi_ for x in r)}
                                               for lo, hi_ in WINDOW_BINS}
        report["groups"][group] = g
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "false_positive_analysis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["groups"], indent=1))


if __name__ == "__main__":
    main()

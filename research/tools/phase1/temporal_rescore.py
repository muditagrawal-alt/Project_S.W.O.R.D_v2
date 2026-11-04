"""
Temporal rescoring of weapon detections in video (Seq-NMS-style, no training).

A detection at frame t with score s and box b looks back k frames; in each it takes the highest-scoring detection
whose centre lies within r * max(w, h) of b's centre (0 if there is none). With m = mean(s, those k scores), the new
score is s' = (1 - lam) * s + lam * m: detections that persist keep their score, one-frame flickers are pulled down.
causal=False also looks k frames ahead (offline reference only; not usable live).

    python research/tools/phase1/temporal_rescore.py --replay <mock-attack replay> --profile cctv-zoom-verified --out <dir>

searches the grid on DEV (Cam1+Cam5), writes every configuration's DEV handgun AP50 and, for the selected causal
configuration, a paired segment bootstrap against no rescoring. --apply k,r,lam additionally scores one configuration on Cam7 and all USRT.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT_ROOT), str(Path(__file__).resolve().parent)]
import cctvgun_usrt_eval as E  # noqa: E402
from ensemble.metrics import precompute_tp  # noqa: E402

GRID = {"k": (1, 2, 4), "r": (0.5, 1.0, 2.0), "lam": (0.25, 0.5, 0.75, 1.0)}


def rescore(frames: list, k: int, r: float, lam: float, causal: bool = True) -> list:
    """frames: per frame a list of (box[4], score). Returns the new scores, same nesting."""
    cen = [np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b, _ in f]).reshape(-1, 2) for f in frames]
    sc = [np.array([s for _, s in f], float) for f in frames]
    out = []
    for t, f in enumerate(frames):
        others = [u for u in range(t - k, t + (0 if causal else k) + 1) if u != t and 0 <= u < len(frames)]
        n_ctx = k if causal else 2 * k   # frames outside the video count as empty, like missed ones
        new = []
        for j, (b, s) in enumerate(f):
            rad = r * max(b[2] - b[0], b[3] - b[1])
            total = s
            for u in others:
                if len(sc[u]):
                    near = np.hypot(*(cen[u] - cen[t][j]).T) <= rad
                    total += sc[u][near].max() if near.any() else 0.0
            new.append((1 - lam) * s + lam * total / (n_ctx + 1))
        out.append(new)
    return out


def segment_frames(replay: Path, f: dict, mode: str) -> dict:
    """Per segment: every recorded frame's kept detections (box, profile score), as cctvgun_usrt_eval.records keeps them."""
    out = {}
    for fj in replay.glob("segments/Cam*/frames.json"):
        d = json.loads(fj.read_text())
        frames = []
        for fr in d["frames"]:
            dets = [(x["bbox"], E.scored(x, f)) for x in fr["weapon_detections"]
                    if E.keep(x, fr["persons"], (d["height"], d["width"]), f) and (mode == "any" or x["cls"] == E.PISTOL)]
            frames.append([(b, s) for b, s in dets if s is not None])
        out[d["video"]] = frames
    return out


def load_index() -> tuple:
    """(annotated images, handgun boxes per image id, image file name -> (segment, frame index))."""
    ann = json.loads(E.ANN.read_text())
    gt = defaultdict(list)
    for a in ann["annotations"]:
        if a["category_id"] == 2:
            x, y, w, h = a["bbox"]
            gt[a["image_id"]].append([x, y, x + w, y + h])
    frame_of = {}
    for sj in E.SEQ.glob("Cam*.json"):
        for i, fr in enumerate(json.loads(sj.read_text()).get("frames", [])):
            frame_of[Path(fr["image"]).name] = (sj.name, i)
    return ann["images"], gt, frame_of


def records(seg_frames: dict, scores: dict | None, index: tuple, cams=None) -> list:
    """CCTV-Gun records (as cctvgun_usrt_eval.records) with optionally replaced scores, optionally for some cameras."""
    images, gt, frame_of = index
    out = []
    for n, img in enumerate(images):
        seg, i = frame_of[img["file_name"]]
        if cams is not None and seg.split("-")[0] not in cams:
            continue
        dets = seg_frames[seg][i]
        s = scores[seg][i] if scores is not None else [x[1] for x in dets]
        g = np.asarray(gt.get(img["id"], []), float).reshape(-1, 4)
        out.append({"image_id": n, "path": img["file_name"], "width": img["width"], "height": img["height"],
                    "segment": seg, "camera": seg.split("-")[0], "gt_cls": np.zeros(len(g), np.int64), "gt_boxes": g,
                    "pred_cls": np.zeros(len(dets), np.int64), "pred_scores": np.asarray(s, float),
                    "pred_boxes": np.asarray([x[0] for x in dets], float).reshape(-1, 4)})
    precompute_tp(out)
    return out


def paired(base: list, cand: list, cams, n_boot: int = 300, seed: int = 0) -> dict:
    """Candidate minus base AP50 on the given cameras, paired segment bootstrap (same resamples)."""
    by_b, by_c = defaultdict(list), defaultdict(list)
    for rb, rc in zip(base, cand):
        if rb["camera"] in cams:
            by_b[rb["segment"]].append(rb)
            by_c[rc["segment"]].append(rc)
    segs = sorted(by_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(segs), len(segs))
        diffs.append(E.ap50([r for i in idx for r in by_c[segs[i]]]) - E.ap50([r for i in idx for r in by_b[segs[i]]]))
    full = E.ap50([r for s in segs for r in by_c[s]]) - E.ap50([r for s in segs for r in by_b[s]])
    return {"diff": round(full, 2), "diff_ci95": [round(float(np.quantile(diffs, q)), 2) for q in (0.025, 0.975)],
            "p_diff_le_0": round(float(np.mean(np.array(diffs) <= 0)), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--profile", default="cctv-zoom-verified")
    ap.add_argument("--mode", default="handgun", choices=("handgun", "any"))
    ap.add_argument("--apply", help="k,r,lam: also score this causal configuration on Cam7 and all USRT")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f = E.filters_of(args.profile)
    sf = segment_frames(Path(args.replay), f, args.mode)
    index = load_index()
    base = records(sf, None, index)
    base_dev = [r for r in base if r["camera"] in E.DEV_CAMERAS]
    dev = lambda recs: round(E.ap50([r for r in recs if r["camera"] in E.DEV_CAMERAS]), 2)  # noqa: E731
    res = {"created": datetime.now().isoformat(timespec="seconds"), "replay": args.replay, "profile": args.profile,
           "mode": args.mode, "grid": GRID, "dev_cameras": E.DEV_CAMERAS, "no_rescoring_dev_ap50": dev(base), "configs": []}
    print("no rescoring, DEV AP50", res["no_rescoring_dev_ap50"], flush=True)
    for causal in (True, False):
        for k, r, lam in itertools.product(*GRID.values()):
            new = {seg: rescore(frames, k, r, lam, causal) for seg, frames in sf.items()}
            row = {"causal": causal, "k": k, "r": r, "lam": lam, "dev_ap50": dev(records(sf, new, index, E.DEV_CAMERAS))}
            res["configs"].append(row)
            print(row, flush=True)
    causal_rows = [c for c in res["configs"] if c["causal"]]
    best = dict(max(causal_rows, key=lambda c: (c["dev_ap50"], -c["k"], -c["lam"])))
    new = {seg: rescore(frames, best["k"], best["r"], best["lam"], True) for seg, frames in sf.items()}
    best.update(paired(base_dev, records(sf, new, index, E.DEV_CAMERAS), E.DEV_CAMERAS, args.n_boot))   # selected only
    res["selected_on_dev"] = best
    print("selected on DEV", best, flush=True)
    res["adopted"] = best["diff_ci95"][0] > 0
    if args.apply:
        k, r, lam = (float(v) for v in args.apply.split(","))
        new = {seg: rescore(frames, int(k), r, lam, True) for seg, frames in sf.items()}
        cand = records(sf, new, index)
        cam7 = lambda recs: round(E.ap50([r for r in recs if r["camera"] == "Cam7"]), 2)  # noqa: E731
        res["held_out"] = {"config": {"k": int(k), "r": r, "lam": lam},
                           "cam7_ap50": cam7(cand), "cam7_ap50_no_rescoring": cam7(base),
                           "cam7_paired": paired(base, cand, ("Cam7",), args.n_boot),
                           "all_usrt_ap50": round(E.ap50(cand), 2), "all_usrt_ap50_no_rescoring": round(E.ap50(base), 2),
                           "all_usrt_ci95": E.cluster_ci(cand, args.n_boot)}
        print("held out", res["held_out"], flush=True)
    (out / ("results_heldout.json" if args.apply else "results_dev.json")).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

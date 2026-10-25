"""
Controlled Phase 1 substitution ("degraded oracle"): does the temporal layer help when the detector has signal?

On real CCTV the S.W.O.R.D. detectors give almost no weapon signal (Phase 1 external evaluation), so the baseline
ladder on recorded runs cannot test the research hypothesis. This replay keeps everything the pipeline recorded
for each mock-attack segment — real tracked persons, real RTMPose keypoints, real PoseC3D action estimates — and
replaces ONLY the weapon detector output with ground-truth boxes degraded in a controlled way:

    misses      burst model (two-state Markov chain shared by the frame): stationary miss rate m, mean burst
                length 2 frames (1 s at 2 fps) — detectors miss in runs, not independently
    jitter      box corners +- N(0, 5% of box size); scores of true detections ~ Beta(5, 2) (mean 0.71)
    false pos.  "transient": Poisson(0.1) per frame at random positions, scores ~ Beta(2, 3)
                "persistent": FP objects (e.g. a phone held in a hand) appear at 2 per minute, last Exp(mean 3 s),
                attached to a recorded person's wrist (or a fixed spot if nobody is tracked), detected in 70% of
                their frames, scores ~ Beta(3, 2) — temporal persistence alone cannot remove these

The synthetic detections go through the SAME weapon tracking + association code as the live pipeline
(tracking.pipeline.weapons_from_detections) and the SAME evaluation (pipeline.run.evaluate_frames). Results are a
simulation of detector quality, not a measurement of the S.W.O.R.D. detectors; parameters are fixed a priori.

    python research/tools/integration/oracle_weapon_replay.py --recording research/runs/<full-pipeline run> \\
        --out research/runs/<id>_oracle-replay
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.rules import FusionConfig  # noqa: E402
from pipeline.run import evaluate_frames  # noqa: E402
from tracking.pipeline import weapons_from_detections  # noqa: E402
from tracking.tracker import ByteTrackWrapper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_full_pipeline import SPLITS, bootstrap, pool  # noqa: E402

SEQ_DIR = PROJECT_ROOT / "research/data/mock_attack/sequences"
MISS_RATES = [0.0, 0.3, 0.6]
FP_MODES = ["none", "transient", "persistent"]
BURST_FRAMES = 2.0
GT_BOX = (37.0, 55.0)  # median GT weapon box (w, h) px in this set


def synth_detections(gt_frames: list, rec_frames: list, miss: float, fp_mode: str, fps: float, size, rng) -> list:
    W, H = size
    p_hm = miss / (BURST_FRAMES * (1 - miss)) if miss < 1 else 1.0   # hit -> miss
    p_mh = 1.0 / BURST_FRAMES                                         # miss -> hit
    state_miss = rng.random() < miss
    fp_objects: list = []   # dicts: until_t, person_id | None, offset/box
    out = []
    for g, r in zip(gt_frames, rec_frames):
        t = g["timestamp"]
        state_miss = (rng.random() >= p_mh) if state_miss else (rng.random() < p_hm)
        boxes, scores, cls = [], [], []
        if not state_miss:
            for b in g["boxes"]:
                x1, y1, x2, y2 = b["bbox"]
                w, h = x2 - x1, y2 - y1
                j = rng.normal(0, 0.05, 4) * [w, h, w, h]
                boxes.append([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3]])
                scores.append(float(rng.beta(5, 2)))
                cls.append(b.get("sword_cls") if b.get("sword_cls") is not None else 0)
        if fp_mode == "transient":
            for _ in range(rng.poisson(0.1)):
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
                boxes.append([cx - GT_BOX[0] / 2, cy - GT_BOX[1] / 2, cx + GT_BOX[0] / 2, cy + GT_BOX[1] / 2])
                scores.append(float(rng.beta(2, 3)))
                cls.append(0)
        elif fp_mode == "persistent":
            if rng.random() < 2.0 / 60.0 / fps:   # 2 objects per minute
                persons = r["persons"]
                pid = int(rng.choice([p["track_id"] for p in persons])) if persons else None
                fp_objects.append({"until": t + rng.exponential(3.0), "pid": pid,
                                   "spot": (rng.uniform(0, W), rng.uniform(0, H)), "hand": int(rng.integers(9, 11))})
            fp_objects = [o for o in fp_objects if t <= o["until"]]
            by_id = {p["track_id"]: p for p in r["persons"]}
            for o in fp_objects:
                if rng.random() > 0.7:
                    continue
                p = by_id.get(o["pid"])
                if p is not None and p["scores"][o["hand"]] > 0.3:
                    cx, cy = p["keypoints"][o["hand"]]
                else:
                    cx, cy = o["spot"]
                boxes.append([cx - GT_BOX[0] / 2, cy - GT_BOX[1] / 2, cx + GT_BOX[0] / 2, cy + GT_BOX[1] / 2])
                scores.append(float(rng.beta(3, 2)))
                cls.append(0)
        out.append((np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float), np.array(cls, dtype=int)))
    return out


def replay_segment(rec_doc: dict, gt_path: Path, miss: float, fp_mode: str, seed: int) -> dict:
    gt = json.loads(gt_path.read_text())
    frames = copy.deepcopy(rec_doc["frames"])
    fps = rec_doc["fps"]
    rng = np.random.default_rng(seed)
    dets = synth_detections(gt["frames"], frames, miss, fp_mode, fps, (rec_doc["width"], rec_doc["height"]), rng)
    trk = ByteTrackWrapper(track_buffer=max(5, int(round(fps * 1.5))))   # as in tracking.pipeline.SwordPipeline
    for f, (b, s, c) in zip(frames, dets):
        f["weapon_detections"], f["weapons"] = weapons_from_detections(trk, b, s, c, f["persons"])
    return evaluate_frames(frames, (rec_doc["height"], rec_doc["width"]), fps, str(gt_path), FusionConfig.for_fps(fps))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True, help="full-pipeline run folder (segments/<name>/frames.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--actions-dir", help="use Phase 4 actions from phase4_replay.py (actions_<segment>.json) instead of the recorded ones")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = {r["segment"]: r for r in json.loads((SEQ_DIR / "index.json").read_text())}
    segs = [s for s in sorted(index) if (Path(args.recording) / "segments" / s / "frames.json").exists()]
    recs = {s: json.loads((Path(args.recording) / "segments" / s / "frames.json").read_text()) for s in segs}
    if args.actions_dir:
        for s in segs:
            acts = json.loads((Path(args.actions_dir) / f"actions_{s}.json").read_text())
            for f, a in zip(recs[s]["frames"], acts):
                assert f["frame_idx"] == a["frame_idx"]
                f["actions"] = a["actions"]
    results, raw = {}, defaultdict(dict)
    for miss, fp_mode in itertools.product(MISS_RATES, FP_MODES):
        cond = f"miss{miss:.1f}_fp-{fp_mode}"
        per_seed = []
        for seed in range(args.seeds):
            evals = {s: replay_segment(recs[s], SEQ_DIR / f"{s}.json", miss, fp_mode, 1000 * seed + i)
                     for i, s in enumerate(segs)}
            raw[cond][seed] = evals
            per_seed.append(evals)
        baselines = [k for k in per_seed[0][segs[0]] if not k.startswith("_")]
        results[cond] = {}
        for split, cams in SPLITS.items():
            names = [s for s in segs if index[s]["camera"] in cams]
            if not names:
                continue
            results[cond][split] = {}
            for b in baselines:
                pooled = [pool([ev[s][b] for s in names]) for ev in per_seed]
                keys = ("verified_event_recall", "verified_delay_s_median", "verified_early_warning_share", "misattributed_alarm_segments",
                        "event_recall", "delay_s_median", "early_warning_share", "false_alarms_per_min_outside_gt",
                        "frame_precision", "frame_recall")
                agg = {k: {"mean": round(float(np.mean([p[k] for p in pooled if p[k] is not None])), 3) if any(p[k] is not None for p in pooled) else None,
                           "sd": round(float(np.std([p[k] for p in pooled if p[k] is not None])), 3) if any(p[k] is not None for p in pooled) else None}
                       for k in keys}
                agg["seed0_ci"] = bootstrap([per_seed[0][s][b] for s in names], n_boot=500)
                agg["gt_events"] = pooled[0]["gt_events"]
                results[cond][split][b] = agg
        print(cond, "| test_cam7  B0 verified recall", results[cond]["test_cam7"]["B0_frame"]["verified_event_recall"]["mean"],
              "FA/min", results[cond]["test_cam7"]["B0_frame"]["false_alarms_per_min_outside_gt"]["mean"],
              "| B4 verified recall", results[cond]["test_cam7"]["B4_rules_full"]["verified_event_recall"]["mean"],
              "FA/min", results[cond]["test_cam7"]["B4_rules_full"]["false_alarms_per_min_outside_gt"]["mean"], flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "per_segment_raw.json").write_text(json.dumps(raw))
    lines = ["# Degraded-oracle replay (simulated Phase 1; real persons, poses and actions)", "",
             "Mean over seeds (sd in brackets). FA/min = false alarms per minute outside ground truth.", ""]
    for split in ("test_cam7", "dev_cam1_cam5", "all"):
        if not all(split in by_split for by_split in results.values()):
            continue
        lines += [f"## {split}", "", "| condition | baseline | VERIFIED event recall | verified delay median s | "
                  "verified early-warning share (isolated) | misattributed alarms | FA/min outside GT | event recall (time-only) | frame P | frame R |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for cond, by_split in results.items():
            for b, m in by_split[split].items():
                f = lambda k: "n/a" if m[k]["mean"] is None else f"{m[k]['mean']} ({m[k]['sd']})"  # noqa: E731
                lines.append(f"| {cond} | {b} | {f('verified_event_recall')} | {f('verified_delay_s_median')} | "
                             f"{f('verified_early_warning_share')} | "
                             f"{f('misattributed_alarm_segments')} | {f('false_alarms_per_min_outside_gt')} | {f('event_recall')} | "
                             f"{f('frame_precision')} | {f('frame_recall')} |")
        lines.append("")
    (out / "results_table.md").write_text("\n".join(lines))
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "recording": str(args.recording), "actions_dir": args.actions_dir, "segments": segs, "seeds": args.seeds, "miss_rates": MISS_RATES,
                "fp_modes": FP_MODES, "burst_frames": BURST_FRAMES, "gt_box_px": GT_BOX,
                "platform": platform.platform(), "python": platform.python_version(),
                "note": "simulation of detector quality; persons/poses/actions are the recorded real pipeline outputs"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

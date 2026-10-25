"""
Re-run Phase 4 (PoseC3D, online) OFFLINE on the skeletons a full-pipeline run recorded, with different options
(e.g. --temporal linear), then re-score: PoseC3D outputs, carrier AUC (mock-attack) and the fusion baseline ladder.
Detectors, tracking and pose are not re-run — only Phase 4 and what depends on it.

    # decision on weapon-free vtest (2 fps vs its own 10 fps reference)
    python research/tools/integration/phase4_replay.py --frames research/runs/<framerate run>/stride5/frames.json \\
        --temporal linear --out <dir>
    # apply to the mock-attack recording
    python research/tools/integration/phase4_replay.py --recording research/runs/<full-pipeline run> --temporal linear --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_recognition.online import OnlineActionRecognizer  # noqa: E402
from fusion.rules import FusionConfig  # noqa: E402
from phase4_carrier_auc import GROUPS, auc_report, carrier_rows  # noqa: E402
from pipeline.run import evaluate_frames  # noqa: E402
from pipeline.system import _compact_action  # noqa: E402
from run_full_pipeline import SPLITS, bootstrap, pool  # noqa: E402
from tracking.pipeline import rederive_weapons  # noqa: E402

SEQ_DIR = PROJECT_ROOT / "research/data/mock_attack/sequences"


def replay(doc: dict, rec: OnlineActionRecognizer) -> list:
    rec.reset()
    frames = doc["frames"]
    shape = (doc["height"], doc["width"], 3)
    for f in frames:
        new = rec.update(f["timestamp"], f["persons"], shape)
        f["actions"] = {int(k): _compact_action(v) for k, v in new.items()}
    return frames


def action_summary(frames: list) -> dict:
    est = [a for f in frames for a in (f.get("actions") or {}).values()]
    out = {"estimates": len(est), "top1": dict(Counter(a["top1_name"] for a in est).most_common(10)),
           "top1_all": dict(Counter(a["top1_name"] for a in est))}
    for g in GROUPS:
        v = np.array([a["group_probs"].get(g, 0.0) for a in est]) if est else np.zeros(1)
        out[g] = {"mean": round(float(v.mean()), 4), "p95": round(float(np.percentile(v, 95)), 4),
                  "share_ge_0.40": round(float((v >= 0.40).mean()), 4), "share_ge_0.55": round(float((v >= 0.55).mean()), 4)}
    return out


def js_divergence(p: dict, q: dict) -> float:
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], float)
    b = np.array([q.get(k, 0) for k in keys], float)
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)
    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", help="full-pipeline run folder (segments/*/frames.json)")
    ap.add_argument("--frames", nargs="*", default=[], help="individual frames.json files")
    ap.add_argument("--reference", help="frames.json whose recorded actions are the reference (JS divergence of top-1)")
    ap.add_argument("--temporal", default="linear", choices=["linear", "nearest"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = OnlineActionRecognizer(device=args.device, temporal=args.temporal)
    jobs = [(Path(f).parent.name, Path(f)) for f in args.frames]
    if args.recording:
        jobs += [(d.name, d / "frames.json") for d in sorted((Path(args.recording) / "segments").iterdir())]
    res = {"temporal": args.temporal, "segments": {}}
    rows, per_seg_eval = [], {}
    all_frames = []
    for name, path in jobs:
        doc = json.loads(path.read_text())
        frames = replay(doc, rec)
        all_frames += frames
        (out / f"actions_{name}.json").write_text(json.dumps([{"frame_idx": f["frame_idx"], "actions": f["actions"]} for f in frames]))
        res["segments"][name] = action_summary(frames)
        gt = SEQ_DIR / f"{name}.json"
        rederive_weapons(frames, doc["fps"])
        ev = evaluate_frames(frames, (doc["height"], doc["width"]), doc["fps"], str(gt) if gt.exists() else None,
                             FusionConfig.for_fps(doc["fps"]))
        per_seg_eval[name] = ev
        if gt.exists():
            rows += carrier_rows(name, frames, json.loads(gt.read_text())["frames"])
        print(f"{name[:40]:40s} {res['segments'][name]['estimates']:5d} estimates | WEAPON_THREAT mean "
              f"{res['segments'][name]['prob_WEAPON_THREAT']['mean']} | B4 alarms/min {ev['B4_rules_full']['alarms_per_min']}", flush=True)
    res["overall_actions"] = action_summary(all_frames)
    if args.reference:
        ref = json.loads(Path(args.reference).read_text())["frames"]
        ref_sum = action_summary(ref)
        res["reference"] = {"file": args.reference, "actions": ref_sum,
                            "js_divergence_top1_vs_reference": round(js_divergence(res["overall_actions"]["top1_all"], ref_sum["top1_all"]), 4)}
    if rows:
        res["carrier_auc"] = auc_report(rows)
    index = {r["segment"]: r for r in json.loads((SEQ_DIR / "index.json").read_text())} if SEQ_DIR.exists() else {}
    res["ladder"] = {}
    for split, cams in SPLITS.items():
        names = [n for n in per_seg_eval if n in index and index[n]["camera"] in cams]
        if names:
            res["ladder"][split] = {b: {**pool([per_seg_eval[n][b] for n in names]), **bootstrap([per_seg_eval[n][b] for n in names], 500)}
                                    for b in per_seg_eval[names[0]] if not b.startswith("_")}
    for n in per_seg_eval:
        if n not in index:
            res["ladder"][n] = {b: {k: v[k] for k in ("alarm_segments", "alarms_per_min", "time_in_alarm_pct")}
                                for b, v in per_seg_eval[n].items() if not b.startswith("_")}
    (out / "phase4_replay.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in res if k in ("reference", "carrier_auc")}, indent=1)[:3000])


if __name__ == "__main__":
    main()

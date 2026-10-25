"""
Does the frame rate explain Phase 4's false "shoot at other person" estimates? Same weapon-free video (vtest.avi,
10 fps), full pipeline at the native rate and losslessly subsampled (every k-th frame), then compares PoseC3D's
group probabilities and the fusion alarms. The mock-attack CCTV set is 2 fps; NTU-120 (PoseC3D training) is 30 fps.

    python research/tools/integration/phase4_framerate.py --out research/runs/<id>_phase4-framerate-vtest --strides 1 2 5
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

from pipeline.run import run_video  # noqa: E402

VTEST = PROJECT_ROOT / "research/data/videos/vtest.avi"
GROUPS = ["prob_WEAPON_THREAT", "prob_VIOLENT_ATTACK", "prob_SUSPICIOUS_AGGRESSION"]


def summarize(run_dir: Path) -> dict:
    doc = json.loads((run_dir / "frames.json").read_text())
    est = [a for f in doc["frames"] for a in (f.get("actions") or {}).values()]
    ev = json.loads((run_dir / "evaluation.json").read_text())
    out = {"fps": doc["fps"], "frames": len(doc["frames"]), "estimates": len(est),
           "top1_names": dict(Counter(a["top1_name"] for a in est).most_common(8)),
           "assigned_groups": dict(Counter(a.get("assigned_group") for a in est))}
    for g in GROUPS:
        v = np.array([a["group_probs"].get(g, 0.0) for a in est])
        out[g] = {"mean": round(float(v.mean()), 4), "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4),
                  "share_ge_0.40": round(float((v >= 0.40).mean()), 4), "share_ge_0.55": round(float((v >= 0.55).mean()), 4)} if len(v) else None
    out["alarms_per_min"] = {k: v["alarms_per_min"] for k, v in ev.items() if not k.startswith("_")}
    out["time_in_alarm_pct"] = {k: v["time_in_alarm_pct"] for k, v in ev.items() if not k.startswith("_")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 5])
    args = ap.parse_args()
    out = Path(args.out)
    res = {}
    for k in args.strides:
        d = out / f"stride{k}"
        if not (d / "evaluation.json").exists():
            run_video(str(VTEST), d, frame_stride=k)
        res[f"stride{k}"] = summarize(d)
        r = res[f"stride{k}"]
        print(f"stride {k} ({r['fps']:.0f} fps): {r['estimates']} estimates | WEAPON_THREAT mean {r['prob_WEAPON_THREAT']['mean']} "
              f"p95 {r['prob_WEAPON_THREAT']['p95']} >=0.55 {r['prob_WEAPON_THREAT']['share_ge_0.55']} | VIOLENT >=0.55 "
              f"{r['prob_VIOLENT_ATTACK']['share_ge_0.55']} | B4 alarms/min {r['alarms_per_min']['B4_rules_full']} | top1 {r['top1_names']}",
              flush=True)
    (out / "phase4_framerate.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

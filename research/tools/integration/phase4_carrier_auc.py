"""
Phase 4 on real CCTV: do PoseC3D's group probabilities separate weapon CARRIERS from other people?

For every online PoseC3D estimate recorded by a full-pipeline run (one per person per second, 3 s window), the
label is 1 if that person carries a ground-truth weapon during the window — carrier = a person the pipeline's own
association links to a ground-truth weapon box (wrist or box containment) in at least one frame of the window.
Reports ROC AUC (with segment-bootstrap CI) of prob_WEAPON_THREAT, prob_VIOLENT_ATTACK, prob_SUSPICIOUS_AGGRESSION
and of the max over the three, plus rates at the fusion thresholds. AUC 0.5 = no information.

    python research/tools/integration/phase4_carrier_auc.py --recording research/runs/<full-pipeline run> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.rules import FusionConfig  # noqa: E402
from tracking.association import associate  # noqa: E402

SEQ_DIR = PROJECT_ROOT / "research/data/mock_attack/sequences"
GROUPS = ["prob_WEAPON_THREAT", "prob_VIOLENT_ATTACK", "prob_SUSPICIOUS_AGGRESSION"]


def auc(scores, labels) -> float | None:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks for ties
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seg_dir in sorted((Path(args.recording) / "segments").iterdir()):
        gt_path = SEQ_DIR / f"{seg_dir.name}.json"
        if gt_path.exists():
            rows += carrier_rows(seg_dir.name, json.loads((seg_dir / "frames.json").read_text())["frames"],
                                 json.loads(gt_path.read_text())["frames"])
    res = auc_report(rows, args.n_boot)
    (out / "phase4_carrier_auc.json").write_text(json.dumps(res, indent=2))
    with open(out / "phase4_estimates.csv", "w") as fp:
        fp.write(",".join(rows[0]) + "\n")
        for r in rows:
            fp.write(",".join(str(v).replace(",", ";") for v in r.values()) + "\n")
    print(json.dumps(res, indent=2))


def carrier_rows(segment: str, frames: list, gt_frames: list) -> list:
    """One row per PoseC3D estimate: group probabilities + whether that person carries a GT weapon in the window."""
    carriers = []
    for f, g in zip(frames, gt_frames):
        links = associate([{"bbox": b["bbox"]} for b in g["boxes"]], f["persons"]) if g["boxes"] else []
        carriers.append({a["person_id"] for a in links if a["person_id"] is not None})
    times = np.array([f["timestamp"] for f in frames])
    rows = []
    for f in frames:
        for tid, a in (f.get("actions") or {}).items():
            tid = int(tid)
            m = (times >= a["window_start"] - 1e-6) & (times <= a["window_end"] + 1e-6)
            label = any(tid in carriers[i] for i in np.nonzero(m)[0])
            gp = a.get("group_probs", {})
            rows.append({"segment": segment, "camera": segment.split("-")[0], "track_id": tid, "t": a["t"],
                         "carrier": bool(label), **{k: float(gp.get(k, 0.0)) for k in GROUPS}, "top1": a.get("top1_name")})
    return rows


def auc_report(rows: list, n_boot: int = 1000) -> dict:
    y = [r["carrier"] for r in rows]
    cfg = FusionConfig()
    res = {"estimates": len(rows), "carrier_estimates": int(sum(y)), "auc": {}, "auc_ci95": {}, "by_camera": {}}
    scores = {k: [r[k] for r in rows] for k in GROUPS}
    scores["max_of_groups"] = [max(r[k] for k in GROUPS) for r in rows]
    by_seg = defaultdict(list)
    for i, r in enumerate(rows):
        by_seg[r["segment"]].append(i)
    segs = sorted(by_seg)
    rng = np.random.default_rng(0)
    for k, s in scores.items():
        res["auc"][k] = auc(s, y)
        boots = []
        for _ in range(n_boot):
            idx = [i for j in rng.integers(0, len(segs), len(segs)) for i in by_seg[segs[j]]]
            v = auc([s[i] for i in idx], [y[i] for i in idx])
            if v is not None:
                boots.append(v)
        res["auc_ci95"][k] = [round(float(np.quantile(boots, 0.025)), 3), round(float(np.quantile(boots, 0.975)), 3)] if boots else None
    for cam in sorted({r["camera"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["camera"] == cam]
        res["by_camera"][cam] = {"estimates": len(idx), "carriers": int(sum(y[i] for i in idx)),
                                 "auc_weapon_threat": auc([scores["prob_WEAPON_THREAT"][i] for i in idx], [y[i] for i in idx])}
    wt = np.array(scores["prob_WEAPON_THREAT"])
    yy = np.array(y)
    for thr in (cfg.a_low, cfg.a_susp):
        res[f"weapon_threat_ge_{thr}"] = {"carrier_rate": round(float((wt[yy] >= thr).mean()), 3) if yy.any() else None,
                                          "non_carrier_rate": round(float((wt[~yy] >= thr).mean()), 3)}
    return res


if __name__ == "__main__":
    main()

"""
Is weapon information present in PoseC3D's NTU-120 output at all? Linear probe on the recorded 120-class
probabilities (no re-run of PoseC3D): logistic regression on log-probabilities averaged over a video's windows.

Mendeley firearm set, location-disjoint split (as everywhere): train P1 + P3, model selection (C) on P2, test P4 + P5.
Tasks: gun vs no-gun, aim vs no-gun, carry vs no-gun, aim vs carry. Reports test AUC with video-level bootstrap CIs
next to the zero-shot WEAPON_THREAT group score on the same test videos, and the most weighted NTU classes.

    python research/tools/integration/phase4_linear_probe.py --recording research/runs/<id>_pipeline-mendeley-stride2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase4_video_eval import boot_auc  # noqa: E402

SPLIT = {"P1": "train", "P3": "train", "P2": "val", "P4": "test", "P5": "test"}
TASKS = {"gun_vs_nogun": (lambda k: k != "no_gun", lambda k: True),
         "aim_vs_nogun": (lambda k: k == "aim", lambda k: k in ("aim", "no_gun")),
         "carry_vs_nogun": (lambda k: k == "carry", lambda k: k in ("carry", "no_gun")),
         "aim_vs_carry": (lambda k: k == "aim", lambda k: k in ("aim", "carry"))}


def load(rec: Path):
    from action_recognition.posec3d import default_action_map_path, load_action_map

    names = load_action_map(default_action_map_path())["all_ntu120_classes"]
    rows = []
    for fj in sorted(rec.glob("videos/*/frames.json")):
        doc = json.loads(fj.read_text())
        est = [a for f in doc["frames"] for a in (f.get("actions") or {}).values() if a.get("probabilities_120")]
        if not est:
            continue
        folder = doc["video"].split("/")[2]
        code, cam, loc = folder.split("_")[:3]
        kind = "aim" if code.startswith("PA") else "carry" if code.startswith("PC") else "no_gun"
        P = np.array([e["probabilities_120"] for e in est], dtype=np.float64)
        rows.append({"video": doc["video"], "kind": kind, "camera": cam, "split": SPLIT[loc],
                     "x": np.log(np.clip(P, 1e-6, 1)).mean(0),
                     "zero_shot": max(e["group_probs"].get("prob_WEAPON_THREAT", 0.0) for e in est)})
    return rows, names


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    args = ap.parse_args()
    rec = Path(args.recording)
    rows, names = load(rec)
    res = {"videos": len(rows), "split_sizes": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}, "tasks": {}}
    for task, (pos, inc) in TASKS.items():
        part = {s: [r for r in rows if r["split"] == s and inc(r["kind"])] for s in ("train", "val", "test")}
        X = {s: np.stack([r["x"] for r in part[s]]) for s in part}
        y = {s: np.array([pos(r["kind"]) for r in part[s]]) for s in part}
        sc = StandardScaler().fit(X["train"])
        best = None
        for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X["train"]), y["train"])
            v = boot_auc(m.predict_proba(sc.transform(X["val"]))[:, 1], y["val"], n=1)["auc"] or 0
            if best is None or v > best[0]:
                best = (v, C, m)
        _, C, m = best
        p_test = m.predict_proba(sc.transform(X["test"]))[:, 1]
        coef = m.coef_[0]
        top = np.argsort(-np.abs(coef))[:8]
        res["tasks"][task] = {
            "C": C, "val_auc": best[0],
            "test_probe": boot_auc(p_test, y["test"]),
            "test_probe_cctv_C1": boot_auc(p_test[[r["camera"] == "C1" for r in part["test"]]],
                                           y["test"][[r["camera"] == "C1" for r in part["test"]]]),
            "test_zero_shot_weapon_threat": boot_auc([r["zero_shot"] for r in part["test"]], y["test"]),
            "top_weighted_ntu_classes": [(names[i], round(float(coef[i]), 3)) for i in top]}
        t = res["tasks"][task]
        print(f"{task:15s} test AUC probe {t['test_probe']['auc']} {t['test_probe']['ci95']} | CCTV C1 {t['test_probe_cctv_C1']['auc']} "
              f"| zero-shot {t['test_zero_shot_weapon_threat']['auc']} {t['test_zero_shot_weapon_threat']['ci95']} | C={C}")
        print("   top classes:", t["top_weighted_ntu_classes"][:5])
    (rec / "phase4_linear_probe.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()

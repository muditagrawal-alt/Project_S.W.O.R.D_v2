"""
Phase 4 (PoseC3D) on surveillance videos at a usable frame rate, from run_video_set.py recordings.

Mendeley firearm set (one person per video, coded action): does PoseC3D separate
    gun vs no-gun videos, aim vs no-gun, carry vs no-gun, and guns vs hard negatives (phone N5-N9, bottle N10-N11)?
Video score = maximum over the video's estimates of a group probability (and the mean). ROC AUC with video-level
bootstrap CIs, overall, per camera (C1 = IP CCTV camera, C2 = phone) and on the declared test locations (P4, P5).
The same videos are re-scored with Phase 4 replayed on a ~2-fps subsample of the recorded skeletons ("nearest" =
repeated frames, "linear" = interpolation) to measure the frame-rate effect with real weapons.

UCF-Crime (annotated test videos): Shooting vs Normal (WEAPON_THREAT), Fighting/Assault vs Normal (VIOLENT_ATTACK),
and inside vs outside the annotated incident window within incident videos.

    python research/tools/integration/phase4_video_eval.py --recording research/runs/<id>_pipeline-mendeley --set mendeley
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase4_carrier_auc import auc  # noqa: E402

GROUPS = ["prob_WEAPON_THREAT", "prob_VIOLENT_ATTACK", "prob_SUSPICIOUS_AGGRESSION"]
HARD_NEG = {f"N{i}" for i in range(5, 12)}
TEST_LOC = {"P4", "P5"}


def estimates(frames: list) -> list:
    return [a for f in frames for a in (f.get("actions") or {}).values()]


def replay_subsampled(doc: dict, every: int, temporal: str, rec) -> list:
    """Phase 4 re-run on every `every`-th recorded frame (skeletons only)."""
    rec.temporal = temporal
    rec.reset()
    shape = (doc["height"], doc["width"], 3)
    out = []
    for i, f in enumerate(doc["frames"]):
        if i % every:
            continue
        for r in rec.update(f["timestamp"], f["persons"], shape).values():
            out.append({"group_probs": r["group_probs"], "top1_name": r["top1_name"], "t": r["t"],
                        "window_start": r["window_start"], "window_end": r["window_end"]})
    return out


def video_scores(est: list) -> dict:
    s = {}
    for g in GROUPS:
        v = [e["group_probs"].get(g, 0.0) for e in est]
        s[f"max_{g}"] = max(v) if v else 0.0
        s[f"mean_{g}"] = float(np.mean(v)) if v else 0.0
    s["n_estimates"] = len(est)
    s["top1"] = Counter(e["top1_name"] for e in est).most_common(3)
    return s


def boot_auc(scores, labels, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores), np.asarray(labels, bool)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        v = auc(s[idx], y[idx])
        if v is not None:
            vals.append(v)
    point = auc(s, y)
    return {"auc": None if point is None else round(point, 3),
            "ci95": [round(float(np.quantile(vals, 0.025)), 3), round(float(np.quantile(vals, 0.975)), 3)] if vals else None,
            "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def mendeley(rec_dir: Path, every_2fps: int, device: str) -> dict:
    from action_recognition.online import OnlineActionRecognizer

    rec = OnlineActionRecognizer(device=device)
    rows = []
    for d in sorted((rec_dir / "videos").iterdir()):
        fj = d / "frames.json"
        if not fj.exists():
            continue
        doc = json.loads(fj.read_text())
        _, cls, folder, _ = doc["video"].split("/")
        code, cam, loc, subj = folder.split("_")[:4]
        kind = "aim" if code.startswith("PA") else "carry" if code.startswith("PC") else "no_gun"
        row = {"video": doc["video"], "class": cls, "code": code, "kind": kind, "camera": cam, "location": loc, "subject": subj,
               "fps": doc["fps"], "native": video_scores(estimates(doc["frames"]))}
        for temporal in ("nearest", "linear"):
            row[f"2fps_{temporal}"] = video_scores(replay_subsampled(doc, every_2fps, temporal, rec))
        rows.append(row)
    res = {"videos": len(rows), "every_2fps": every_2fps, "comparisons": {}}
    comparisons = {
        "gun_vs_nogun": (lambda r: r["kind"] != "no_gun", lambda r: True),
        "aim_vs_nogun": (lambda r: r["kind"] == "aim", lambda r: r["kind"] in ("aim", "no_gun")),
        "carry_vs_nogun": (lambda r: r["kind"] == "carry", lambda r: r["kind"] in ("carry", "no_gun")),
        "gun_vs_hard_negatives": (lambda r: r["kind"] != "no_gun", lambda r: r["kind"] != "no_gun" or r["code"] in HARD_NEG),
        "aim_vs_carry": (lambda r: r["kind"] == "aim", lambda r: r["kind"] in ("aim", "carry")),
    }
    subsets = {"all": lambda r: True, "cctv_C1": lambda r: r["camera"] == "C1", "phone_C2": lambda r: r["camera"] == "C2",
               "test_locations_P4_P5": lambda r: r["location"] in TEST_LOC}
    for cname, (is_pos, include) in comparisons.items():
        for sname, sub in subsets.items():
            rs = [r for r in rows if include(r) and sub(r)]
            if not rs:
                continue
            y = [is_pos(r) for r in rs]
            for variant in ("native", "2fps_nearest", "2fps_linear"):
                for stat in ("max", "mean"):
                    for g in GROUPS:
                        key = f"{cname}|{sname}|{variant}|{stat}_{g}"
                        res["comparisons"][key] = boot_auc([r[variant][f"{stat}_{g}"] for r in rs], y)
    res["top1_by_kind_native"] = {k: Counter(t for r in rows if r["kind"] == k for t, _ in r["native"]["top1"][:1]).most_common(6)
                                  for k in ("aim", "carry", "no_gun")}
    res["per_video"] = rows
    return res


def ucf(rec_dir: Path) -> dict:
    ann_file = next((PROJECT_ROOT / "research/data/ucf_crime/annotations").rglob("Temporal_Anomaly_Annotation.txt"))
    ann = {r.split()[0]: r.split() for r in ann_file.read_text().splitlines() if r.strip()}
    rows, windows = [], defaultdict(list)
    for d in sorted((rec_dir / "videos").iterdir()):
        fj = d / "frames.json"
        if not fj.exists():
            continue
        doc = json.loads(fj.read_text())
        name = doc["video"].split("/")[-1]
        if name not in ann:
            continue   # training videos (no temporal annotation)
        cat = ann[name][1]
        f_native = native_fps(doc)   # annotations are native frame numbers; timestamps are seconds
        segs = [(int(a) / f_native, int(b) / f_native) for a, b in zip(ann[name][2::2], ann[name][3::2]) if int(a) >= 0]
        est = estimates(doc["frames"])
        rows.append({"video": doc["video"], "category": cat, **video_scores(est)})
        for e in est:
            inside = any(a <= e["window_end"] <= b for a, b in segs)
            windows[cat].append((inside, e["group_probs"]))
    res = {"videos": len(rows), "comparisons": {}}
    for cname, pos, g in (("shooting_vs_normal", {"Shooting"}, "prob_WEAPON_THREAT"),
                          ("robbery_vs_normal", {"Robbery"}, "prob_WEAPON_THREAT"),
                          ("fight_assault_vs_normal", {"Fighting", "Assault"}, "prob_VIOLENT_ATTACK")):
        rs = [r for r in rows if r["category"] in pos | {"Normal"}]
        if rs:
            res["comparisons"][f"{cname}|video|max_{g}"] = boot_auc([r[f"max_{g}"] for r in rs], [r["category"] in pos for r in rs])
            inside = [(i, p[g]) for c in pos for i, p in windows[c]]
            if inside:
                res["comparisons"][f"{cname}|windows_inside_vs_outside_incident|{g}"] = boot_auc([p for _, p in inside], [i for i, _ in inside])
    res["per_video"] = rows
    return res


def native_fps(doc: dict) -> float:
    """Recordings store the processed fps; incident annotations are in native frames."""
    import cv2

    path = PROJECT_ROOT / "research/data/ucf_crime/extracted/UCF_Crimes/Videos" / "/".join(doc["video"].split("/")[1:])
    cap = cv2.VideoCapture(str(path))
    f = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--set", required=True, choices=["mendeley", "ucf"])
    ap.add_argument("--every-2fps", type=int, default=6, help="subsample factor of the recording for the ~2-fps variant")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    rec = Path(args.recording)
    res = mendeley(rec, args.every_2fps, args.device) if args.set == "mendeley" else ucf(rec)
    out = rec / f"phase4_eval_{args.set}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    key = [k for k in res["comparisons"] if "|all|" in k or args.set == "ucf"]
    for k in key[:60]:
        if "max_prob_WEAPON" in k or "VIOLENT" in k or args.set == "ucf":
            print(k, res["comparisons"][k])
    print("->", out)


if __name__ == "__main__":
    main()

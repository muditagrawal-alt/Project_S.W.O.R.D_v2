"""
Train the cross-fitted weapon-action head (action_recognition/weapon_head.py) on PoseC3D outputs recorded for the
Mendeley firearm set, and evaluate it on held-out locations.

Samples: one per recorded PoseC3D estimate (3 s window). Positives: windows of gun videos (aim/carry/reload) that
overlap the boxed weapon span (derived labels); negatives: all windows of no-gun videos (walking, running, phone,
bottle, heavy object). Gun-video windows outside the boxed span are excluded (weapon possibly not visible).

Cross-fitting (no score is ever produced by a model trained on that video):
  - C chosen by leave-one-location-out over the training locations P1, P2, P3 (window-level AUC)
  - out-of-fold heads: P1 <- head(P2, P3), P2 <- head(P1, P3), P3 <- head(P1, P2)
  - deployment head: trained on P1 + P2 + P3; used for the test locations P4, P5, for UCF-Crime and the mock-attack set
Calibration (pre-declared, from out-of-fold no-gun windows of P1-P3): l_low = 95th, l_susp = 99th percentile.

    python research/tools/phase4/train_weapon_head.py --recording research/runs/<id>_pipeline-mendeley-stride2 \\
        --out research/runs/<id>_weapon-head
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/integration"))

from action_recognition.weapon_head import DEFAULT_HEAD, WeaponActionHead, save_head  # noqa: E402
from fusion.labels import by_video, load_labels  # noqa: E402
from phase4_carrier_auc import auc  # noqa: E402
from phase4_video_eval import boot_auc  # noqa: E402

TRAIN_LOCS, TEST_LOCS = ("P1", "P2", "P3"), ("P4", "P5")
CS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)   # extended down: 0.001 was the grid edge


def lowfps_estimates(doc: dict, every: int, recognizer) -> list:
    """PoseC3D re-run on every `every`-th recorded frame (skeletons only) -> estimates at ~fps/every."""
    recognizer.reset()
    shape = (doc["height"], doc["width"], 3)
    out = []
    for i, f in enumerate(doc["frames"]):
        if i % every:
            continue
        for tid, r in recognizer.update(f["timestamp"], f["persons"], shape).items():
            out.append((f["frame_idx"], tid, {"probabilities_120": r["probabilities_120"], "window_start": r["window_start"],
                                              "window_end": r["window_end"], "t": r["t"]}))
    return out


def load_windows(rec: Path, labels: dict, lowfps_every: int = 0, device: str = "mps") -> list:
    rows = []
    recognizer = None
    if lowfps_every:
        from action_recognition.online import OnlineActionRecognizer

        recognizer = OnlineActionRecognizer(device=device, weapon_head=None)
    for fj in sorted(rec.glob("videos/*/frames.json")):
        doc = json.loads(fj.read_text())
        vid = doc["video"]
        folder = vid.split("/")[2]
        code, cam, loc = folder.split("_")[:3]
        gun = not code.startswith("N")
        span = [(iv.start_s, iv.end_s) for iv in labels.get(vid, []) if iv.label == "WEAPON_RELATED_EVENT"]
        estimates = [(f["frame_idx"], int(tid), a, "native") for f in doc["frames"] for tid, a in (f.get("actions") or {}).items()]
        if recognizer is not None:   # augmentation: the same skeletons at low frame rate (real CCTV is often ~2 fps)
            estimates += [(fi, tid, a, "lowfps") for fi, tid, a in lowfps_estimates(doc, lowfps_every, recognizer)]
        for frame_idx, tid, a, variant in estimates:
            if not a.get("probabilities_120"):
                continue
            if gun and not any(a["window_end"] >= s and a["window_start"] <= e for s, e in span):
                continue
            rows.append({"video": vid, "frame_idx": frame_idx, "track_id": int(tid), "t": a["t"], "variant": variant,
                         "loc": loc, "camera": cam, "code": code, "y": int(gun),
                         "kind": "aim" if code.startswith("PA") else "carry" if code.startswith("PC") else "no_gun",
                         "x": WeaponActionHead.features(a["probabilities_120"])})
    return rows


def fit(rows, C):
    from sklearn.linear_model import LogisticRegression

    X = np.stack([r["x"] for r in rows])
    y = np.array([r["y"] for r in rows])
    mean, std = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit((X - mean) / std, y)
    return mean, std, m.coef_[0], float(m.intercept_[0])


def score(head, rows):
    mean, std, coef, b = head
    X = np.stack([r["x"] for r in rows])
    return 1 / (1 + np.exp(-(((X - mean) / std) @ coef + b)))


def video_auc(rows, scores, agg="max"):
    by = defaultdict(list)
    lab = {}
    for r, s in zip(rows, scores):
        by[r["video"]].append(s)
        lab[r["video"]] = r["y"]
    vids = sorted(by)
    v = [max(by[k]) if agg == "max" else float(np.mean(by[k])) for k in vids]
    return boot_auc(v, [lab[k] for k in vids])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True, help="pipeline run over the Mendeley videos (videos/*/frames.json)")
    ap.add_argument("--labels", default=str(PROJECT_ROOT / "research/data/labels/mendeley_derived.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--install", action="store_true", help=f"also write the deployment head to {DEFAULT_HEAD}")
    ap.add_argument("--augment-lowfps", type=int, default=0, metavar="EVERY",
                    help="also train on PoseC3D re-run on every EVERY-th recorded frame (e.g. 6: 12.5 fps -> ~2 fps)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_windows(Path(args.recording), by_video(load_labels(Path(args.labels))), args.augment_lowfps, args.device)
    tr = [r for r in rows if r["loc"] in TRAIN_LOCS]
    te = [r for r in rows if r["loc"] in TEST_LOCS]
    print(f"windows: train locations {len(tr)} (pos {sum(r['y'] for r in tr)}), test locations {len(te)} (pos {sum(r['y'] for r in te)})")

    # 1) C by leave-one-location-out on P1-P3
    cv = {}
    for C in CS:
        aucs = []
        for held in TRAIN_LOCS:
            a = [r for r in tr if r["loc"] != held]
            b = [r for r in tr if r["loc"] == held]
            aucs.append(auc(score(fit(a, C), b), [r["y"] for r in b]))
        cv[C] = float(np.mean(aucs))
    C = max(cv, key=cv.get)
    print("LOLO window AUC by C:", {k: round(v, 3) for k, v in cv.items()}, "-> C =", C)

    # 2) out-of-fold heads and scores for P1-P3
    oof = np.zeros(len(tr))
    for held in TRAIN_LOCS:
        idx = [i for i, r in enumerate(tr) if r["loc"] == held]
        head = fit([r for r in tr if r["loc"] != held], C)
        oof[idx] = score(head, [tr[i] for i in idx])
        save_head(out / f"fold_head_{held}.json", *head, C=C, version="v1-fold", heldout_location=held,
                  trained_on_locations=[l for l in TRAIN_LOCS if l != held])

    # 3) calibration on out-of-fold no-gun windows
    neg = oof[np.array([r["y"] for r in tr]) == 0]
    cal = {"l_low": float(np.quantile(neg, 0.95)), "l_susp": float(np.quantile(neg, 0.99)),
           "rule": "95th / 99th percentile of out-of-fold no-gun window scores (Mendeley P1-P3)"}
    print("calibration:", {k: round(v, 3) if isinstance(v, float) else v for k, v in cal.items()})

    # 4) deployment head on P1+P2+P3, evaluated on the held-out test locations
    dep = fit(tr, C)
    meta = dict(C=C, version="v1" + ("-lowfps" if args.augment_lowfps else ""), augment_lowfps_every=args.augment_lowfps, features="log NTU-120 probabilities (PoseC3D SlowOnly-R50, linear temporal resampling)",
                trained_on="Mendeley firearm set, locations P1+P2+P3 (Ruiz-Santaquiteria et al. 2024, CC BY-NC 3.0)",
                calibration=cal, created=datetime.now().isoformat(timespec="seconds"))
    save_head(out / "posec3d_weapon_head_v1.json", *dep, **meta)
    if args.install:
        save_head(DEFAULT_HEAD, *dep, **meta)
    te_all = te
    te = [r for r in te_all if r.get("variant", "native") == "native"]   # headline test metrics on native windows (comparable)
    s_te = score(dep, te)
    y_te = [r["y"] for r in te]
    res = {"C": C, "lolo_auc_by_C": cv, "calibration": cal, "n_windows": {"train": len(tr), "test": len(te)},
           "oof_window_auc_P1_P3": auc(oof, [r["y"] for r in tr]),
           "test": {"window_auc": boot_auc(s_te, y_te), "video_auc_max": video_auc(te, s_te, "max"),
                    "video_auc_mean": video_auc(te, s_te, "mean")}}
    for cam in ("C1", "C2"):
        sub = [i for i, r in enumerate(te) if r["camera"] == cam]
        res["test"][f"window_auc_{cam}"] = boot_auc(s_te[sub], [y_te[i] for i in sub])
    for kind in ("aim", "carry"):
        sub = [i for i, r in enumerate(te) if r["kind"] in (kind, "no_gun")]
        res["test"][f"window_auc_{kind}_vs_nogun"] = boot_auc(s_te[sub], [y_te[i] for i in sub])
    low = [r for r in te_all if r.get("variant") == "lowfps"]
    if low:
        res["test"]["window_auc_lowfps"] = boot_auc(score(dep, low), [r["y"] for r in low])
    fpr_low = float(np.mean(s_te[np.array(y_te) == 0] >= cal["l_low"]))
    tpr_low = float(np.mean(s_te[np.array(y_te) == 1] >= cal["l_low"]))
    fpr_s = float(np.mean(s_te[np.array(y_te) == 0] >= cal["l_susp"]))
    tpr_s = float(np.mean(s_te[np.array(y_te) == 1] >= cal["l_susp"]))
    res["test"]["at_l_low"] = {"tpr": round(tpr_low, 3), "fpr": round(fpr_low, 3)}
    res["test"]["at_l_susp"] = {"tpr": round(tpr_s, 3), "fpr": round(fpr_s, 3)}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    # cross-fitted scores for every Mendeley window (out-of-fold for P1-P3, deployment head for P4-P5)
    xfit = [{"video": r["video"], "frame_idx": r["frame_idx"], "track_id": r["track_id"], "score": float(s)}
            for r, s in list(zip(tr, oof)) + list(zip(te, s_te)) if r.get("variant", "native") == "native"]
    (out / "crossfit_scores_mendeley.json").write_text(json.dumps(xfit))
    print(json.dumps(res["test"], indent=1, default=float))


if __name__ == "__main__":
    main()

"""
Tune the Phase 5 operating point (rules + zoom-detection threshold) on DEV data only; report it on held-out data.

DEV (selection):
  - mock-attack Cam1 + Cam5: verified event recall of the alarm rung (level >= SUSPICIOUS_ACTIVITY) and false alarms
    per minute outside weapon events
  - Mendeley validation location P2: alarms per minute on no-gun clips
  - UCF-Crime normal videos of the transformer's training half (seeded split): alarms per minute in crowded,
    weapon-free real CCTV
TEST (reporting only): mock-attack Cam7, weapon-free vtest, Mendeley P4 + P5, UCF-Crime test half (minus the videos
the detector was trained on).

Searched: rule parameters (RULE_GRID: weapon-evidence thresholds, persistence, hold time, and the thresholds at which
the learned weapon-action head and PoseC3D's violent-attack group alone raise SUSPICIOUS_ACTIVITY) x the zoom-crop score
threshold. A first run (v1: weapon thresholds only) showed that on weapon-free real CCTV most alarms come from the
action channels, which v1 could not reach.
Fixed a priori from the TRAINING data, not tuned (FIXED_FILTERS): boxes larger than 12.5% of the frame are dropped (the
largest CCTV training box is 12.3% of its frame; research/data/cctv_det, web images excluded) and zoom windows smaller
than 96 px are skipped (the smallest training zoom crop). Their effect is reported as an ablation.

Objective (declared before running): maximise DEV verified recall subject to the DEV false-alarm rates <= the budget;
ties -> lower total DEV false alarms, then closer to the defaults. If nothing meets the budget, the configuration with
the lowest worst-case DEV false-alarm rate is reported as infeasible. v3 (--objective pooled, default) applies the budget
to mock-attack dev and to all weapon-free dev minutes pooled (~36 min): v1/v2 applied it per source, and the Mendeley P2
no-gun set (2.2 min, 2 alarms in every trial) alone made every configuration infeasible. Because the trade-off between
recall and false alarms is a curve, one operating point is selected per budget (--fa-budget 0.5 1.0) and each is
reported on the held-out data; the DEV frontier is saved (dev_tradeoff.png).

    python research/tools/fusion/tune_rules.py --usrt <replay> --mendeley <replay> --ucf <replay> --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing as mp
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/integration"))

from fusion.evaluate import evaluate_alarms, replay_rules, segments, verified_metrics  # noqa: E402
from fusion.labels import by_video, load_labels, scene_levels  # noqa: E402
from fusion.rules import LEVELS, FusionConfig  # noqa: E402
from run_full_pipeline import pool  # noqa: E402
from tracking.pipeline import rederive_weapons  # noqa: E402

SEQ = PROJECT_ROOT / "research/data/mock_attack/sequences"
LABELS = PROJECT_ROOT / "research/data/labels"
ZOOM_MIN = [0.1, 0.2, 0.3, 0.4]
FIXED_FILTERS = {"max_area_frac": 0.125, "zoom_min_window": 96}   # a priori from the training data (see above)
FILTERS_OFF = {"max_area_frac": 1.0, "zoom_min_window": 0}
# second-stage verifier (tracking/verifier.py; needs "verify" in the recordings): (fusion mode, minimum p(weapon))
VERIFY = [("none", 0.0), ("geo", 0.0), ("product", 0.0), ("geo", 0.3)]
VERIFY_OFF = {"verify_mode": "none", "verify_min": 0.0}
DET_KEYS = ("zoom_min_score", "max_area_frac", "zoom_min_window", "verify_mode", "verify_min")
RULE_GRID = {"w_susp": [0.3, 0.4, 0.5], "w_event": [0.5, 0.6], "persist_s": [0.5, 1.0, 1.5], "hold_s": [1.0, 2.0],
             "l_susp": [0.725, 0.8, 0.9, 1.01],   # 0.725 = calibrated 99th no-gun percentile; 1.01 = head never alarms alone
             "a_susp": [0.55, 0.7, 0.85]}         # 0.55 = default
DETECTOR_OVERLAP = {"Robbery106", "Shooting022", "Shooting028", "Shooting008", "Shooting004", "Shooting011",
                    "Shooting024", "Shooting013", "Shooting018", "Shooting026", "Shooting046"}


def load(run: Path, pattern: str) -> list:
    return [json.loads(fj.read_text()) for fj in sorted(run.glob(pattern))]


def ucf_split(docs: list) -> tuple:
    """Same seeded 50/50 split by category as research/tools/fusion/train_eval_transformer.py."""
    by_cat = defaultdict(list)
    for d in docs:
        by_cat[d["video"].split("/")[1]].append(d)
    rng, train, test = random.Random(0), [], []
    for cat, ds in sorted(by_cat.items()):
        ds = sorted(ds, key=lambda d: d["video"])
        rng.shuffle(ds)
        train += ds[: len(ds) // 2]
        test += ds[len(ds) // 2:]
    return train, test


def cfg_for(base: FusionConfig, params: dict, fps: float) -> FusionConfig:
    cfg = copy.deepcopy(base)
    for k, v in params.items():
        if k not in DET_KEYS:
            setattr(cfg, k, v)
    cfg.grace_s = max(0.6, 2.0 / fps + 1e-3)
    return cfg


def alarm_flags(d, cfg) -> list:
    thr = LEVELS.index("SUSPICIOUS_ACTIVITY")
    return [LEVELS.index(x) >= thr for x in replay_rules(d["frames"], (d["height"], d["width"]), cfg)]


def apply_det(docs: list, params: dict, zoom_modes: tuple) -> list:
    """Copies of the recordings with the Phase 1 detection filters of `params` applied (tracking/pipeline.py)."""
    docs = copy.deepcopy(docs)
    for d in docs:
        rederive_weapons(d["frames"], d["fps"], frame_shape=(d["height"], d["width"]), zoom_modes=zoom_modes,
                         **{k: params[k] for k in DET_KEYS})
    return docs


_DEV = {}


def _init_worker(dev: dict):
    _DEV.update(dev)


def dev_trial(dev: dict, params: dict) -> dict:
    base = FusionConfig()
    u = usrt_eval(dev["usrt_dev"], base, params)
    (ma, mm), (ua, um) = (weapon_free_counts(dev[k], base, params) for k in ("men_dev_nogun", "ucf_dev_normal"))
    return {"params": params, "dev_verified_recall": u["verified_event_recall"],
            "dev_fa_usrt": u["false_alarms_per_min_outside_gt"],
            "dev_fa_mendeley_nogun": round(ma / max(mm, 1e-9), 3), "dev_fa_ucf_normal": round(ua / max(um, 1e-9), 3),
            "dev_fa_weapon_free_pooled": round((ma + ua) / max(mm + um, 1e-9), 3), "dev_weapon_free_min": round(mm + um, 2)}


def det_trials(det: dict, fixed_rules: dict, zoom_modes: tuple) -> list:
    """All remaining rule combinations for one detection setting and one value of the first rule keys (a worker task)."""
    dev = {name: apply_det(docs, det, zoom_modes) for name, docs in _DEV.items()}
    keys = [k for k in RULE_GRID if k not in fixed_rules]
    out = []
    for values in itertools.product(*(RULE_GRID[k] for k in keys)):
        params = dict(zip(keys, values), **fixed_rules, **det)
        if params["w_event"] < params["w_susp"]:
            continue
        out.append(dev_trial(dev, params))
    return out


def usrt_eval(docs, base, params) -> dict:
    return pool(usrt_segment_evals(docs, base, params))


def usrt_segment_evals(docs, base, params) -> list:
    """Per-segment alarm + verified metrics of the alarm rung (raw counts, for pooling and the segment bootstrap)."""
    evals = []
    for d in docs:
        cfg = cfg_for(base, params, d["fps"])
        gt = json.loads((SEQ / d["video"]).read_text())["frames"][: len(d["frames"])]
        flags_gt = [bool(f["boxes"]) for f in gt]
        times = [f["timestamp"] for f in d["frames"]]
        mg = max(1.0, 2.5 / d["fps"])
        gsegs = segments(times, flags_gt, mg)
        lv, tr = replay_rules(d["frames"], (d["height"], d["width"]), cfg, with_tracks=True)
        thr = LEVELS.index("SUSPICIOUS_ACTIVITY")
        det = {"r": {"flags": [LEVELS.index(x) >= thr for x in lv], "ids": [{k for k, v in t.items() if v >= thr} for t in tr]}}
        ev = evaluate_alarms(times, det["r"]["flags"], gsegs, flags_gt, merge_gap_s=mg)
        ev.update(verified_metrics(d["frames"], det, [[b["bbox"] for b in f["boxes"]] for f in gt], gsegs, merge_gap_s=mg)["r"])
        evals.append(ev)
    return evals


def weapon_free_counts(docs, base, params) -> tuple:
    """(alarm segments, minutes) on footage without any weapon event (every alarm is false)."""
    alarms, minutes = 0, 0.0
    for d in docs:
        times = [f["timestamp"] for f in d["frames"]]
        ev = evaluate_alarms(times, alarm_flags(d, cfg_for(base, params, d["fps"])))
        alarms += ev["alarm_segments"]
        minutes += ev["duration_min"]
    return alarms, minutes


def weapon_free_rate(docs, base, params) -> float:
    alarms, minutes = weapon_free_counts(docs, base, params)
    return round(alarms / max(minutes, 1e-9), 3)


def scene_eval(docs, labels, base, params) -> dict:
    agg = defaultdict(float)
    for d in docs:
        if d["video"] not in labels:
            continue
        t = np.array([f["timestamp"] for f in d["frames"]])
        y = scene_levels(labels[d["video"]], t)
        a = np.array(alarm_flags(d, cfg_for(base, params, d["fps"])))
        ev = evaluate_alarms(t, a, segments(t, y >= 2, 1.0), y >= 2, merge_gap_s=1.0)
        for k in ("gt_events", "events_detected", "false_alarm_segments", "negative_min", "duration_min", "alarm_segments"):
            agg[k] += ev.get(k) or 0
    return {"event_recall": round(agg["events_detected"] / max(agg["gt_events"], 1), 3), "events": int(agg["gt_events"]),
            "fa_per_min": round(agg["false_alarm_segments"] / max(agg["negative_min"], 1e-9), 3),
            "alarms_per_min": round(agg["alarm_segments"] / max(agg["duration_min"], 1e-9), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usrt", required=True, help="mock-attack replay (segments/*/frames.json, incl. vtest)")
    ap.add_argument("--mendeley", required=True, help="Mendeley replay (videos/*/frames.json)")
    ap.add_argument("--ucf", required=True, help="UCF-Crime replay (videos/*/frames.json)")
    ap.add_argument("--fa-budget", type=float, nargs="+", default=[0.5, 1.0],
                    help="false-alarm budgets (per minute); one dev-selected operating point each")
    ap.add_argument("--reuse-trials", help="trials.json of a run with the same grid and recordings (selection only)")
    ap.add_argument("--verifier", action="store_true", help="also search the second-stage verifier fusion (VERIFY); the "
                                                            "recordings must carry 'verify' (apply_weapon_verifier.py)")
    ap.add_argument("--objective", choices=("pooled", "per-source"), default="pooled",
                    help="pooled: the budget applies to mock-attack dev and to ALL weapon-free dev minutes together; "
                         "per-source (v1/v2): to each dev source (the 2-minute Mendeley P2 no-gun set then decides alone)")
    ap.add_argument("--zoom-modes", default="hands", help="zoom modes the replays were recorded with")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    segs = {d["video"]: d for d in load(Path(args.usrt), "segments/*/frames.json")}
    men = load(Path(args.mendeley), "videos/*/frames.json")
    ucf = [d for d in load(Path(args.ucf), "videos/*/frames.json")
           if d["video"].split("/")[-1].replace("_x264.mp4", "") not in DETECTOR_OVERLAP]
    men_labels = by_video(load_labels(LABELS / "mendeley_derived.csv"))
    ucf_labels = by_video(load_labels(LABELS / "ucf_derived.csv"))
    loc = lambda d: d["video"].split("/")[2].split("_")[2]  # noqa: E731
    ucf_tr, ucf_te = ucf_split([d for d in ucf if d["video"] in ucf_labels])
    sets = {"usrt_dev": [d for k, d in segs.items() if k.startswith(("Cam1", "Cam5"))],
            "usrt_test": [d for k, d in segs.items() if k.startswith("Cam7")],
            "vtest": [d for k, d in segs.items() if k == "vtest.avi"],
            "men_dev_nogun": [d for d in men if loc(d) == "P2" and d["video"].split("/")[1] == "No_Gun"],
            "men_test": [d for d in men if loc(d) in ("P4", "P5")],
            "ucf_dev_normal": [d for d in ucf_tr if "Normal" in d["video"].split("/")[1]],
            "ucf_test": ucf_te}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    base = FusionConfig()
    keys = list(RULE_GRID)
    zoom_modes = tuple(m for m in args.zoom_modes.split("+") if m)
    verify = VERIFY if args.verifier else [("none", 0.0)]
    dets = [{"zoom_min_score": z, **FIXED_FILTERS, "verify_mode": m, "verify_min": vm}
            for z in (ZOOM_MIN if zoom_modes else [0.1]) for m, vm in verify]
    split_keys = ("l_susp", "a_susp")   # one worker task per detection setting x (l_susp, a_susp)
    tasks = [(det, dict(zip(split_keys, v))) for det in dets for v in itertools.product(*(RULE_GRID[k] for k in split_keys))]
    dev_sets = {name: sets[name] for name in ("usrt_dev", "men_dev_nogun", "ucf_dev_normal")}
    if args.reuse_trials:   # same grid and recordings, only the selection changes
        trials = json.loads(Path(args.reuse_trials).read_text())
        print(f"reusing {len(trials)} trials from {args.reuse_trials}", flush=True)
    else:
        trials = []
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                                 initargs=(dev_sets,)) as ex:
            for i, res in enumerate(ex.map(det_trials, [t[0] for t in tasks], [t[1] for t in tasks], [zoom_modes] * len(tasks))):
                trials += res
                print(f"  task {i + 1}/{len(tasks)} {tasks[i]}: {len(trials)} trials so far", flush=True)
    fa_keys = (("dev_fa_usrt", "dev_fa_weapon_free_pooled") if args.objective == "pooled"
               else ("dev_fa_usrt", "dev_fa_mendeley_nogun", "dev_fa_ucf_normal"))
    default_params = {k: getattr(base, k) for k in keys} | {"zoom_min_score": 0.1, **FILTERS_OFF, **VERIFY_OFF}
    dist = lambda t: sum(abs(t["params"][k] - v) if isinstance(v, (int, float)) else float(t["params"][k] != v)  # noqa: E731
                         for k, v in default_params.items())
    worst = lambda t: max(t[k] for k in fa_keys)  # noqa: E731

    def select(budget):
        feasible = [t for t in trials if worst(t) <= budget]
        if feasible:
            return max(feasible, key=lambda t: (t["dev_verified_recall"], -sum(t[k] for k in fa_keys), -dist(t))), len(feasible)
        return min(trials, key=lambda t: (worst(t), -t["dev_verified_recall"])), 0

    def report(params):
        res = {}
        for name in ("usrt_test", "vtest", "men_test", "ucf_test"):
            docs = apply_det(sets[name], params, zoom_modes)
            if name == "usrt_test":
                res[name] = usrt_eval(docs, base, params)
            elif name == "vtest":
                res["vtest_alarms_per_min"] = weapon_free_rate(docs, base, params)
            else:
                res[name] = scene_eval(docs, men_labels if name == "men_test" else ucf_labels, base, params)
        return res

    # DEV trade-off: configurations not beaten on both dev verified recall and worst dev false-alarm rate
    frontier = sorted({(t["dev_verified_recall"], worst(t)) for t in trials if not any(
        o["dev_verified_recall"] >= t["dev_verified_recall"] and worst(o) <= worst(t)
        and (o["dev_verified_recall"], worst(o)) != (t["dev_verified_recall"], worst(t)) for o in trials)}, key=lambda x: x[1])
    operating_points = {}
    for budget in args.fa_budget:
        best, n_feasible = select(budget)
        print(f"budget {budget}: selected on DEV {best} | feasible: {n_feasible}", flush=True)
        operating_points[str(budget)] = {"selected": best, "n_feasible": n_feasible, "test": report(best["params"]),
                                         "test_filters_off": report(best["params"] | FILTERS_OFF),
                                         **({"test_verifier_off": report(best["params"] | VERIFY_OFF)} if args.verifier else {})}
    first = operating_points[str(args.fa_budget[0])]
    dev_full = {name: apply_det(docs, default_params, zoom_modes) for name, docs in dev_sets.items()}
    default_rules_filters = default_params | FIXED_FILTERS | {"zoom_min_score": first["selected"]["params"]["zoom_min_score"]}
    result = {"created": datetime.now().isoformat(timespec="seconds"), "fa_budgets": args.fa_budget, "objective": args.objective,
              "fa_keys": fa_keys, "grid": {"zoom_min_score": ZOOM_MIN, **RULE_GRID, "verify": verify}, "fixed_filters": FIXED_FILTERS,
              "set_sizes": {k: len(v) for k, v in sets.items()}, "zoom_modes": zoom_modes, "n_trials": len(trials),
              "reused_trials": args.reuse_trials, "dev_frontier_recall_vs_worst_fa": frontier,
              "default_on_dev": dev_trial(dev_full, default_params), "test_default": report(default_params),
              "test_default_rules_fixed_filters": report(default_rules_filters),
              "operating_points": operating_points,
              "selected": first["selected"], "test_selected": first["test"]}   # first budget, for install_profile.py
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([worst(t) for t in trials], [t["dev_verified_recall"] for t in trials], s=6, alpha=0.3, label="configurations")
        ax.plot([f[1] for f in frontier], [f[0] for f in frontier], "-o", ms=3, color="C1", label="DEV frontier")
        d = result["default_on_dev"]
        ax.scatter([max(d[k] for k in fa_keys)], [d["dev_verified_recall"]], marker="x", s=60, color="k", label="defaults")
        for b, op in operating_points.items():
            ax.scatter([worst(op["selected"])], [op["selected"]["dev_verified_recall"]], marker="*", s=120, label=f"selected, budget {b}")
        ax.set_xlabel("worst DEV false alarms per minute (" + ", ".join(k.replace("dev_fa_", "") for k in fa_keys) + ")")
        ax.set_ylabel("DEV verified event recall (mock-attack Cam1+Cam5)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "dev_tradeoff.png", dpi=150)
    except Exception as e:   # noqa: BLE001
        print("figure skipped:", e)
    (out / "trials.json").write_text(json.dumps(trials, indent=1))
    (out / "results.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({b: {"dev": {k: v for k, v in op["selected"].items() if k != "params"}, "params": op["selected"]["params"]}
                      for b, op in operating_points.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()

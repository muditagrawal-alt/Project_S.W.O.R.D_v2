"""
Train the scene-level Temporal Threat Transformer on interval labels and compare it with the rules engine and with
frame-level detection on held-out data, using identical inputs (pipeline recordings) and metrics.

Splits (declared before training):
  mendeley  train = locations P1 + P3, val = P2, test = P4 + P5 (same as the detector fine-tuning set)
  ucf       labelled UCF-Crime test videos split 50/50 by video, stratified by category, seed 0 (train / test)
  usrt      mock-attack CCTV: external test only
Model selection: epoch count by validation macro-F1 (checked every 5 epochs); 3 seeds, mean ± sd reported.

Metrics per test set: 5 Hz step-level macro-F1 and per-level F1 (unknown steps ignored); alarm metrics with
fusion.evaluate.evaluate_alarms where alarm = predicted level >= SUSPICIOUS_ACTIVITY and ground-truth events =
labelled intervals >= SUSPICIOUS_ACTIVITY (event recall, delay, false alarms/min outside events).

    python research/tools/fusion/train_eval_transformer.py --mendeley research/runs/<id>_pipeline-mendeley \\
        --ucf research/runs/<id>_pipeline-ucf --usrt research/runs/<id>_full-pipeline-... --out research/runs/<id>_transformer
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.evaluate import evaluate_alarms, replay_rules, segments  # noqa: E402
from fusion.labels import by_video, load_labels, scene_levels  # noqa: E402
from fusion.rules import LEVELS, FusionConfig  # noqa: E402
from fusion.transformer import (SCENE_FEATURES, TemporalThreatTransformer, WindowDataset, export_scene_sequences,  # noqa: E402
                                predict_steps, save_model, scene_windows)
from tracking.pipeline import rederive_weapons  # noqa: E402

LABELS = PROJECT_ROOT / "research" / "data" / "labels"
MENDELEY_SPLIT = {"P1": "train", "P3": "train", "P2": "val", "P4": "test", "P5": "test"}
RATE, WIN = 5.0, 50


def load_set(rec_dirs, label_csv: Path, manifest: Path):
    labels = by_video(load_labels(label_csv))
    exh = json.loads(manifest.read_text()).get("extra", {}).get("exhaustive_per_video", {})
    covers = json.loads(manifest.read_text()).get("covers_video", True)
    items = []
    for rec in rec_dirs:
        for fj in sorted(list(Path(rec).glob("videos/*/frames.json")) + list(Path(rec).glob("segments/*/frames.json"))):
            doc = json.loads(fj.read_text())
            if doc["video"] not in labels:
                continue
            rederive_weapons(doc["frames"], doc["fps"])   # current association code
            ts, x = export_scene_sequences(doc, RATE)
            lv = scene_levels(labels[doc["video"]], ts, exhaustive=exh.get(doc["video"], covers))
            items.append({"video": doc["video"], "doc": doc, "ts": ts, "x": x, "y": lv})
    return items


def rules_steps(item, cfg: FusionConfig) -> np.ndarray:
    doc = item["doc"]
    lv = replay_rules(doc["frames"], (doc["height"], doc["width"]), cfg)
    ft = np.array([f["timestamp"] for f in doc["frames"]])
    idx = np.clip(np.searchsorted(ft, item["ts"], side="right") - 1, 0, len(ft) - 1)
    return np.array([LEVELS.index(lv[i]) for i in idx])


def b0_steps(item, conf: float = 0.5) -> np.ndarray:
    raw = item["x"][:, SCENE_FEATURES.index("max_raw_detection")]
    return np.where(raw >= conf, 3, 0)


def f1_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = y_true >= 0
    yt, yp = y_true[m], y_pred[m]
    out = {}
    for k, name in enumerate(LEVELS):
        tp = int(((yp == k) & (yt == k)).sum())
        fp = int(((yp == k) & (yt != k)).sum())
        fn = int(((yp != k) & (yt == k)).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[name] = {"f1": round(2 * p * r / max(p + r, 1e-9), 3), "support": int((yt == k).sum())}
    present = [v["f1"] for v in out.values() if v["support"] > 0]
    return {"macro_f1": round(float(np.mean(present)), 3) if present else None, "accuracy": round(float((yt == yp).mean()), 3), "per_level": out}


def alarm_report(items, preds) -> dict:
    agg = defaultdict(float)
    delays = []
    for it, p in zip(items, preds):
        known = it["y"] >= 0
        if not known.any():
            continue
        ts = it["ts"][known]
        gt = it["y"][known] >= 2
        ev = evaluate_alarms(ts, p[known] >= 2, segments(ts, gt, 1.0), gt, merge_gap_s=1.0)
        for k in ("duration_min", "alarm_segments", "gt_events", "events_detected", "false_alarm_segments", "negative_min"):
            agg[k] += ev.get(k) or 0
        delays += ev.get("delays_s", [])
    return {"event_recall": round(agg["events_detected"] / max(agg["gt_events"], 1), 3), "gt_events": int(agg["gt_events"]),
            "false_alarms_per_min_outside_events": round(agg["false_alarm_segments"] / max(agg["negative_min"], 1e-9), 3),
            "delay_s_median": round(float(np.median(delays)), 2) if delays else None}


def evaluate_methods(items, model=None) -> dict:
    cfg_cache = {}
    preds = {"B0_frame": [b0_steps(it) for it in items]}
    rules = []
    for it in items:
        fps = it["doc"]["fps"]
        cfg = cfg_cache.setdefault(round(fps, 3), FusionConfig.for_fps(fps))
        rules.append(rules_steps(it, copy.deepcopy(cfg)))
    preds["rules_B4"] = rules
    if model is not None:
        preds["transformer"] = [predict_steps(model, it["x"], WIN).argmax(1) for it in items]
    y = np.concatenate([it["y"] for it in items]) if items else np.zeros(0, int)
    return {name: {**f1_report(y, np.concatenate(p)), "alarms": alarm_report(items, p)} for name, p in preds.items()}


def train_select(train_items, val_items, seed: int, max_epochs: int, device: str):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    win_items = [w for it in train_items for w in scene_windows(it["x"], it["y"], WIN, stride=2)]
    counts = np.bincount([w[2] for w in win_items], minlength=len(LEVELS)).astype(float)
    weights = counts.sum() / np.maximum(counts, 1) / len(LEVELS)
    weights[counts == 0] = 0
    model = TemporalThreatTransformer(n_features=len(SCENE_FEATURES)).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    loader = torch.utils.data.DataLoader(WindowDataset(win_items), batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    best = (-1.0, 0, None)
    history = []
    for ep in range(1, max_epochs + 1):
        model.train()
        for x, m, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), m.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep % 5 == 0 or ep == max_epochs:
            yv = np.concatenate([it["y"] for it in val_items])
            pv = np.concatenate([predict_steps(model, it["x"], WIN, device).argmax(1) for it in val_items])
            f1 = f1_report(yv, pv)["macro_f1"] or 0.0
            history.append({"epoch": ep, "val_macro_f1": f1})
            if f1 > best[0]:
                best = (f1, ep, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    return model.eval(), {"best_epoch": best[1], "best_val_macro_f1": best[0], "history": history,
                          "train_windows": len(win_items), "train_class_counts": counts.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mendeley", nargs="+", required=True)
    ap.add_argument("--ucf", nargs="*", default=[])
    ap.add_argument("--usrt", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    men = load_set(args.mendeley, LABELS / "mendeley_derived.csv", LABELS / "mendeley_derived.manifest.json")
    split = lambda it: MENDELEY_SPLIT[it["video"].split("/")[2].split("_")[2]]  # noqa: E731
    sets = {"mendeley_train": [i for i in men if split(i) == "train"], "mendeley_val": [i for i in men if split(i) == "val"],
            "mendeley_test": [i for i in men if split(i) == "test"]}
    if args.ucf:
        ucf = load_set(args.ucf, LABELS / "ucf_derived.csv", LABELS / "ucf_derived.manifest.json")
        by_cat = defaultdict(list)
        for it in ucf:
            by_cat[it["video"].split("/")[1]].append(it)
        sets["ucf_train"], sets["ucf_test"] = [], []
        rng = random.Random(0)
        for cat, its in sorted(by_cat.items()):
            its = sorted(its, key=lambda i: i["video"])
            rng.shuffle(its)
            sets["ucf_train"] += its[: len(its) // 2]
            sets["ucf_test"] += its[len(its) // 2:]
    if args.usrt:
        sets["usrt_external_test"] = load_set(args.usrt, LABELS / "usrt_derived_TEST_ONLY.csv",
                                              LABELS / "usrt_derived_TEST_ONLY.manifest.json")
    train_items = sets["mendeley_train"] + sets.get("ucf_train", [])
    results = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
               "set_sizes": {k: len(v) for k, v in sets.items()}, "features": SCENE_FEATURES, "rate_hz": RATE, "win": WIN,
               "label_sources": "derived (unverified) — see fusion/LABELING_GUIDELINES.md", "seeds": {}}
    tests = [k for k in sets if k.endswith("test")]
    baselines = {k: evaluate_methods(sets[k]) for k in tests}
    for seed in range(args.seeds):
        model, info = train_select(train_items, sets["mendeley_val"], seed, args.epochs, args.device)
        results["seeds"][seed] = {"training": info, "test": {k: evaluate_methods(sets[k], model)["transformer"] for k in tests}}
        save_model(model, out / f"transformer_scene_seed{seed}.pt", SCENE_FEATURES, {"rate_hz": RATE, "win": WIN, **info})
        print(f"seed {seed}: best epoch {info['best_epoch']} val macro-F1 {info['best_val_macro_f1']}", flush=True)
    results["baselines"] = baselines
    summary = {}
    for k in tests:
        tf = [results["seeds"][s]["test"][k] for s in results["seeds"]]
        summary[k] = {"transformer_macro_f1": f"{np.mean([t['macro_f1'] for t in tf]):.3f} ± {np.std([t['macro_f1'] for t in tf]):.3f}",
                      "transformer_event_recall": f"{np.mean([t['alarms']['event_recall'] for t in tf]):.3f}",
                      "transformer_fa_per_min": f"{np.mean([t['alarms']['false_alarms_per_min_outside_events'] for t in tf]):.3f}",
                      "rules_macro_f1": baselines[k]["rules_B4"]["macro_f1"], "rules_event_recall": baselines[k]["rules_B4"]["alarms"]["event_recall"],
                      "rules_fa_per_min": baselines[k]["rules_B4"]["alarms"]["false_alarms_per_min_outside_events"],
                      "b0_macro_f1": baselines[k]["B0_frame"]["macro_f1"], "b0_event_recall": baselines[k]["B0_frame"]["alarms"]["event_recall"],
                      "b0_fa_per_min": baselines[k]["B0_frame"]["alarms"]["false_alarms_per_min_outside_events"]}
    results["summary"] = summary
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

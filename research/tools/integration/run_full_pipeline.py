"""
Full-pipeline test (Phases 1, 2-stand-in, 3, 4, 5): every mock-attack segment with ground truth + weapon-free vtest.

For one detector configuration, runs pipeline.run on each sequence, then pools the baseline ladder (B0 frame-level
detection ... B4 temporal rules with actions) across segments per pre-declared split:
    dev_cam1_cam5  (any configuration choice is made here)    test_cam7  (held-out cameras)    all
    vtest          weapon-free street footage: every alarm is a false alarm
Pooled metrics: event recall (+ segment-bootstrap CI), detection delay (median, share of early warnings),
false alarms per minute outside ground truth (+ CI), frame precision/recall, time in alarm. Also diagnostics for
each phase (tracking fragmentation, pose coverage, action-group distribution, weapon association) and timings.

    python research/tools/integration/run_full_pipeline.py --name yolo26s-640 --out research/runs/<id>_full-pipeline-yolo26s-640
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.rules import FusionConfig  # noqa: E402
from pipeline.run import evaluate_frames, run_video  # noqa: E402
from tracking.pipeline import rederive_weapons  # noqa: E402
from pipeline.system import SwordSystem  # noqa: E402

SPLITS = {"dev_cam1_cam5": ("Cam1", "Cam5"), "test_cam7": ("Cam7",), "all": ("Cam1", "Cam5", "Cam7")}
SEQ_DIR = PROJECT_ROOT / "research/data/mock_attack/sequences"
VTEST = PROJECT_ROOT / "research/data/videos/vtest.avi"


def diagnostics(frames_json: Path, gt_flags, frames: list | None = None) -> dict:
    frames = frames if frames is not None else json.loads(frames_json.read_text())["frames"]
    seen = defaultdict(list)
    pose_ok = person_frames = 0
    for f in frames:
        for p in f["persons"]:
            seen[p["track_id"]].append(f["timestamp"])
            person_frames += 1
            pose_ok += max(p["scores"]) > 0
    lengths = [len(v) for v in seen.values()]
    durations = [v[-1] - v[0] for v in seen.values()]
    groups = {"weapon_time": Counter(), "other_time": Counter()}
    for i, f in enumerate(frames):
        key = "weapon_time" if gt_flags and gt_flags[i] else "other_time"
        for a in (f.get("actions") or {}).values():
            groups[key][a.get("assigned_group") or "NONE"] += 1
    wdet = [w for f in frames for w in f["weapons"]]
    return {"person_tracks": len(seen), "person_frames": person_frames,
            "track_length_frames_median": float(np.median(lengths)) if lengths else 0.0,
            "track_duration_s_median": round(float(np.median(durations)), 2) if durations else 0.0,
            "single_frame_tracks_share": round(float(np.mean([n == 1 for n in lengths])), 3) if lengths else 0.0,
            "pose_coverage": round(pose_ok / max(person_frames, 1), 3),
            "action_updates": sum(len(f.get("actions") or {}) for f in frames),
            "action_groups": {k: dict(v) for k, v in groups.items()},
            "weapon_obs": len(wdet), "weapon_confirmed_track_share": round(np.mean([w["track_id"] is not None for w in wdet]), 3) if wdet else None,
            "weapon_linked_share": round(np.mean([w["person_id"] is not None for w in wdet]), 3) if wdet else None,
            "weapon_wrist_share": round(np.mean([w["assoc_method"] == "wrist" for w in wdet]), 3) if wdet else None,
            "raw_weapon_detections_ge_0.5": sum(d["score"] >= 0.5 for f in frames for d in f["weapon_detections"])}


def pool(evals: list) -> dict:
    """Pool per-segment evaluate_alarms outputs (each with raw counts)."""
    out = {"segments": len(evals), "duration_min": round(sum(e["duration_min"] for e in evals), 2),
           "alarm_segments": sum(e["alarm_segments"] for e in evals)}
    out["alarms_per_min"] = round(out["alarm_segments"] / max(out["duration_min"], 1e-9), 3)
    if evals and "gt_events" in evals[0]:
        ev = sum(e["gt_events"] for e in evals)
        det = sum(e["events_detected"] for e in evals)
        delays = [d for e in evals for d in e["delays_s"]]
        iso = [d for e in evals for d in e.get("isolated_delays_s", [])]
        fa = sum(e["false_alarm_segments"] for e in evals)
        neg = sum(e["negative_min"] for e in evals)
        tp, al, gt = (sum(e[k] for e in evals) for k in ("frame_tp", "frame_alarms", "frame_gt"))
        out.update(gt_events=ev, events_detected=det, event_recall=round(det / max(ev, 1), 3),
                   delay_s_median=round(float(np.median(delays)), 2) if delays else None,
                   isolated_events=sum(e.get("isolated_events", 0) for e in evals),
                   early_warning_share=round(float(np.mean([d < 0 for d in iso])), 3) if iso else None,  # isolated events only
                   false_alarm_segments=fa, negative_min=round(neg, 2), false_alarms_per_min_outside_gt=round(fa / max(neg, 1e-9), 3),
                   frame_precision=round(tp / max(al, 1), 3), frame_recall=round(tp / max(gt, 1), 3))
        if "verified_events_detected" in evals[0]:
            vdet = sum(e["verified_events_detected"] for e in evals)
            vdel = [d for e in evals for d in e["verified_delays_s"]]
            viso = [d for e in evals for d in e.get("verified_isolated_delays_s", [])]
            out.update(verified_events_detected=vdet, verified_event_recall=round(vdet / max(ev, 1), 3),
                       verified_delay_s_median=round(float(np.median(vdel)), 2) if vdel else None,
                       verified_early_warning_share=round(float(np.mean([d < 0 for d in viso])), 3) if viso else None,
                       misattributed_alarm_segments=sum(e["misattributed_alarm_segments"] for e in evals))
    return out


def bootstrap(evals: list, n_boot: int = 1000, seed: int = 0) -> dict:
    if not evals or "gt_events" not in evals[0]:
        return {}
    rng = np.random.default_rng(seed)
    rec, far, vrec = [], [], []
    for _ in range(n_boot):
        sample = [evals[i] for i in rng.integers(0, len(evals), len(evals))]
        p = pool(sample)
        rec.append(p["event_recall"])
        far.append(p["false_alarms_per_min_outside_gt"])
        vrec.append(p.get("verified_event_recall", np.nan))
    q = lambda v: [round(float(np.nanquantile(v, 0.025)), 3), round(float(np.nanquantile(v, 0.975)), 3)]  # noqa: E731
    out = {"event_recall_ci95": q(rec), "false_alarms_per_min_ci95": q(far), "unit": "segment", "n_boot": n_boot}
    if not np.all(np.isnan(vrec)):
        out["verified_event_recall_ci95"] = q(vrec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="configuration label, e.g. yolo26s-640")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weapon-model", default=str(PROJECT_ROOT / "weights/yolo26s_best.pt"))
    ap.add_argument("--weapon-imgsz", type=int, default=640)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--segments", nargs="*", help="segment names (default: all)")
    ap.add_argument("--no-vtest", action="store_true")
    ap.add_argument("--save-video", nargs="*", default=[], help="segments to also render annotated.mp4 for")
    ap.add_argument("--actions-dir", help="with --reevaluate: use Phase 4 actions from phase4_replay.py "
                                          "(actions_<segment>.json) instead of the recorded ones")
    ap.add_argument("--reevaluate", action="store_true",
                    help="re-score already recorded segments from frames.json with the current association/fusion/evaluation code "
                         "(the recorded evaluation is kept as evaluation_recorded.json)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fp = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        log_fp.write(line + "\n")
        log_fp.flush()

    index = {r["segment"]: r for r in json.loads((SEQ_DIR / "index.json").read_text())}
    segs = args.segments or sorted(index)
    kw = dict(weapon_model=args.weapon_model, weapon_imgsz=args.weapon_imgsz, use_ensemble=args.ensemble)
    systems = {}

    def system_for(fps):
        if fps not in systems:
            systems[fps] = SwordSystem(fps=fps, **kw)
        return systems[fps]

    per_seg, rows = {}, []
    jobs = [(s, SEQ_DIR / f"{s}.json", True) for s in segs] + ([] if args.no_vtest else [("vtest", VTEST, False)])
    t_all = time.time()
    for name, src, has_gt in jobs:
        seg_dir = out / "segments" / name
        rederived = None
        if (seg_dir / "evaluation.json").exists():
            res = {"summary": json.loads((seg_dir / "summary.json").read_text()),
                   "evaluation": json.loads((seg_dir / "evaluation.json").read_text())}
            if args.reevaluate:
                if not (seg_dir / "evaluation_recorded.json").exists():
                    (seg_dir / "evaluation.json").rename(seg_dir / "evaluation_recorded.json")
                doc = json.loads((seg_dir / "frames.json").read_text())
                if args.actions_dir:
                    acts = json.loads((Path(args.actions_dir) / f"actions_{name}.json").read_text())
                    for f, a in zip(doc["frames"], acts):
                        assert f["frame_idx"] == a["frame_idx"]
                        f["actions"] = a["actions"]
                rederive_weapons(doc["frames"], doc["fps"])   # current tracking/association code
                rederived = doc["frames"]
                cfg = FusionConfig.for_fps(doc["fps"])
                res["evaluation"] = evaluate_frames(doc["frames"], (doc["height"], doc["width"]), doc["fps"],
                                                    str(src) if has_gt else None, cfg)
                (seg_dir / "evaluation.json").write_text(json.dumps(res["evaluation"], indent=2))
                log(f"[re-evaluated] {name}")
            else:
                log(f"[skip] {name} (already done)")
        else:
            fps = 2.0 if has_gt else 10.0
            log(f"{name}: {index[name]['frames'] if has_gt else 795} frames ...")
            res = run_video(str(src), seg_dir, system=system_for(fps), gt=str(src) if has_gt else None,
                            save_video=name in args.save_video)
        gt_flags = None
        if has_gt:
            doc = json.loads(src.read_text())
            gt_flags = [bool(f["boxes"]) for f in doc["frames"]]
        diag = diagnostics(seg_dir / "frames.json", gt_flags, rederived)
        (seg_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        per_seg[name] = {"camera": index[name]["camera"] if has_gt else "vtest", "evaluation": res["evaluation"],
                         "timing_ms": res["summary"]["timing_ms"], "frames": res["summary"]["frames"], "diagnostics": diag}
        b0, b4 = res["evaluation"]["B0_frame"], res["evaluation"]["B4_rules_full"]
        log(f"  {name}: B0 alarms {b0['alarm_segments']} recall {b0.get('event_recall')} | B4 alarms {b4['alarm_segments']} "
            f"recall {b4.get('event_recall')} | {res['summary']['timing_ms'].get('throughput_fps')} fps | tracks {diag['person_tracks']}")
        for base, e in res["evaluation"].items():
            if base.startswith("_"):
                continue
            rows.append({"segment": name, "camera": per_seg[name]["camera"], "baseline": base,
                         **{k: e.get(k) for k in ("duration_min", "alarm_segments", "alarms_per_min", "time_in_alarm_pct",
                                                  "gt_events", "events_detected", "event_recall", "detection_delay_s_median",
                                                  "false_alarm_segments", "false_alarms_per_min_outside_gt",
                                                  "frame_precision", "frame_recall")}})

    baselines = [k for k in next(iter(per_seg.values()))["evaluation"] if not k.startswith("_")]
    results = {"config": args.name, "detector": kw, "splits": {}}
    for split, cams in SPLITS.items():
        names = [n for n, v in per_seg.items() if v["camera"] in cams]
        if not names:
            continue
        results["splits"][split] = {b: {**pool([per_seg[n]["evaluation"][b] for n in names]),
                                        **bootstrap([per_seg[n]["evaluation"][b] for n in names])} for b in baselines}
    if "vtest" in per_seg:
        results["splits"]["vtest_weapon_free"] = {b: pool([per_seg["vtest"]["evaluation"][b]]) for b in baselines}
    # timings weighted by frames
    tot = defaultdict(float)
    n = 0
    for v in per_seg.values():
        for k, t in v["timing_ms"].items():
            if isinstance(t, dict):
                tot[k] += t["mean"] * v["frames"]
        n += v["frames"]
    results["timing_ms_mean"] = {k: round(x / n, 2) for k, x in tot.items()}
    results["throughput_fps"] = round(1000.0 / results["timing_ms_mean"]["total_ms"], 2)
    agg = defaultdict(list)
    for name, v in per_seg.items():
        if name != "vtest":
            for k in ("person_tracks", "track_duration_s_median", "single_frame_tracks_share", "pose_coverage"):
                agg[k].append(v["diagnostics"][k])
    results["diagnostics_mockattack"] = {"person_tracks_total": int(sum(agg["person_tracks"])),
                                         "track_duration_s_median_of_segments": float(np.median(agg["track_duration_s_median"])),
                                         "single_frame_tracks_share_mean": round(float(np.mean(agg["single_frame_tracks_share"])), 3),
                                         "pose_coverage_mean": round(float(np.mean(agg["pose_coverage"])), 3)}
    groups = {"weapon_time": Counter(), "other_time": Counter()}
    for name, v in per_seg.items():
        if name != "vtest":
            for k in groups:
                groups[k].update(v["diagnostics"]["action_groups"][k])
    results["action_groups_mockattack"] = {k: dict(v) for k, v in groups.items()}
    results["per_segment"] = per_seg
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    with open(out / "per_segment.csv", "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    lines = [f"# Full pipeline — {args.name}", "",
             f"Detector: `{Path(args.weapon_model).name if not args.ensemble else 'WBF ensemble'}` @ imgsz {args.weapon_imgsz}; "
             f"throughput {results['throughput_fps']} fps (mean over all frames).", ""]
    for split, table in results["splits"].items():
        lines += [f"## {split}", ""]
        if split == "vtest_weapon_free":
            lines += ["| Baseline | alarms | alarms/min | time in alarm % |", "|---|---|---|---|"]
            for b, m in table.items():
                tia = per_seg["vtest"]["evaluation"][b]["time_in_alarm_pct"]
                lines.append(f"| {b} | {m['alarm_segments']} | {m['alarms_per_min']} | {tia} |")
        else:
            lines += ["| Baseline | VERIFIED event recall (95% CI) | verified delay median s | verified early-warning share (isolated events) | "
                      "event recall, time-only | misattributed alarms | FA/min outside GT (95% CI) | frame P (time) | frame R (time) |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for b, m in table.items():
                lines.append(f"| {b} | {m.get('verified_events_detected')}/{m['gt_events']} = {m.get('verified_event_recall')} "
                             f"{m.get('verified_event_recall_ci95', '')} | {m.get('verified_delay_s_median')} | "
                             f"{m.get('verified_early_warning_share')} (n={m.get('isolated_events')}) | "
                             f"{m['events_detected']}/{m['gt_events']} = {m['event_recall']} | {m.get('misattributed_alarm_segments')} | "
                             f"{m['false_alarms_per_min_outside_gt']} {m.get('false_alarms_per_min_ci95', '')} | "
                             f"{m['frame_precision']} | {m['frame_recall']} |")
        lines.append("")
    (out / "results_table.md").write_text("\n".join(lines))
    git = lambda c: subprocess.run(["git", *c], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "args": vars(args), "git_head": git(["rev-parse", "HEAD"]), "git_dirty": bool(git(["status", "--porcelain"])),
                "segments": list(per_seg), "wall_minutes": round((time.time() - t_all) / 60, 1),
                "sources": next(iter(systems.values())).sources if systems else None,
                "platform": platform.platform(), "python": platform.python_version(),
                "reevaluated_with_current_code": args.reevaluate, "actions_dir": args.actions_dir,
                "notes": ["mock-attack frames are 2 fps; FusionConfig.for_fps adapts the persistence grace (a priori)",
                          "fusion action thresholds were calibrated on vtest only; nothing was tuned on mock-attack"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log("\n" + "\n".join(lines))
    for s in systems.values():
        s.close()


if __name__ == "__main__":
    main()

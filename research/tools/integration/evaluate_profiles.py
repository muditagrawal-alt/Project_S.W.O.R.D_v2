"""
Compare system profiles (pipeline/profiles.py) on the same recordings, with the same metrics as the dev tuner
(research/tools/fusion/tune_rules.py): each profile is scored on recordings replayed with ITS detector (persons,
poses and Phase 4 actions are shared), with its detection filters and fusion settings applied.

Reported per profile, alarm rung = rules level >= SUSPICIOUS_ACTIVITY:
  DEV   mock-attack Cam1+Cam5 verified recall / false alarms per minute outside weapon events; Mendeley P2 no-gun and
        UCF-Crime dev-half normal videos: alarms per minute (all false)
  TEST  mock-attack Cam7 (verified, segment-bootstrap CI), weapon-free vtest alarms per minute, Mendeley P4+P5 and
        UCF-Crime test half (derived interval labels): event recall and false alarms per minute

    python research/tools/integration/evaluate_profiles.py --out <dir> \\
        --profile original:<usrt replay>:<mendeley replay>:<ucf replay> --profile cctv-zoom:<...>:<...>:<...>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/fusion"))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/integration"))

from fusion.labels import by_video, load_labels  # noqa: E402
from fusion.rules import FusionConfig  # noqa: E402
from pipeline.profiles import list_profiles  # noqa: E402
from run_full_pipeline import bootstrap, pool  # noqa: E402
from tune_rules import (DETECTOR_OVERLAP, LABELS, apply_det, load, scene_eval, ucf_split, usrt_eval,  # noqa: E402
                        usrt_segment_evals, weapon_free_rate)

TRIAL_KEYS = ("w_susp", "w_event", "persist_s", "hold_s")


def profile_params(p: dict) -> dict:
    """The profile as tune_rules parameters (rule values default to FusionConfig's)."""
    base = FusionConfig()
    fusion = p.get("fusion", {})
    return {**{k: fusion.get(k, getattr(base, k)) for k in TRIAL_KEYS}, **{k: v for k, v in fusion.items() if k not in TRIAL_KEYS},
            "zoom_min_score": float(p.get("zoom_min_score", 0.0)), "max_area_frac": float(p.get("weapon_max_area", 1.0)),
            "zoom_min_window": int(p.get("zoom_min_window", 0))}


def sets_for(usrt: Path, mendeley: Path, ucf: Path) -> dict:
    segs = {d["video"]: d for d in load(usrt, "segments/*/frames.json")}
    men = load(mendeley, "videos/*/frames.json")
    ucf_docs = [d for d in load(ucf, "videos/*/frames.json")
                if d["video"].split("/")[-1].replace("_x264.mp4", "") not in DETECTOR_OVERLAP]
    ucf_labels = by_video(load_labels(LABELS / "ucf_derived.csv"))
    loc = lambda d: d["video"].split("/")[2].split("_")[2]  # noqa: E731
    ucf_tr, ucf_te = ucf_split([d for d in ucf_docs if d["video"] in ucf_labels])
    return {"usrt_dev": [d for k, d in segs.items() if k.startswith(("Cam1", "Cam5"))],
            "usrt_test": [d for k, d in segs.items() if k.startswith("Cam7")],
            "vtest": [d for k, d in segs.items() if k == "vtest.avi"],
            "men_dev_nogun": [d for d in men if loc(d) == "P2" and d["video"].split("/")[1] == "No_Gun"],
            "men_test": [d for d in men if loc(d) in ("P4", "P5")],
            "ucf_dev_normal": [d for d in ucf_tr if "Normal" in d["video"].split("/")[1]],
            "ucf_test": ucf_te}


def evaluate(sets: dict, params: dict, zoom_modes: tuple) -> dict:
    """Profile parameters other than the tuned rule keys (e.g. l_low) are applied through cfg_for as well."""
    base = FusionConfig()
    men_labels = by_video(load_labels(LABELS / "mendeley_derived.csv"))
    ucf_labels = by_video(load_labels(LABELS / "ucf_derived.csv"))
    d = {name: apply_det(docs, params, zoom_modes) for name, docs in sets.items()}
    dev = usrt_eval(d["usrt_dev"], base, params)
    test_evals = usrt_segment_evals(d["usrt_test"], base, params)
    test, ci = pool(test_evals), bootstrap(test_evals)
    return {"params": params,
            "dev": {"usrt_verified_recall": dev["verified_event_recall"], "usrt_fa_per_min": dev["false_alarms_per_min_outside_gt"],
                    "mendeley_nogun_alarms_per_min": weapon_free_rate(d["men_dev_nogun"], base, params),
                    "ucf_normal_alarms_per_min": weapon_free_rate(d["ucf_dev_normal"], base, params)},
            "test": {"usrt": {k: test.get(k) for k in ("verified_event_recall", "event_recall", "false_alarms_per_min_outside_gt",
                                                        "misattributed_alarm_segments", "gt_events", "alarm_segments")},
                     "usrt_ci": ci, "vtest_alarms_per_min": weapon_free_rate(d["vtest"], base, params),
                     "mendeley": scene_eval(d["men_test"], men_labels, base, params),
                     "ucf": scene_eval(d["ucf_test"], ucf_labels, base, params)}}


def table(results: dict) -> str:
    head = ("| profile | dev recall (USRT) | dev FA/min USRT · Mendeley · UCF | Cam7 verified recall | Cam7 FA/min | "
            "vtest alarms/min | Mendeley test recall · FA/min | UCF test recall · FA/min |")
    rows = [head, "|" + "---|" * 8]
    for name, r in results.items():
        dv, t = r["dev"], r["test"]
        rows.append(f"| {name} | {dv['usrt_verified_recall']} | {dv['usrt_fa_per_min']} · {dv['mendeley_nogun_alarms_per_min']} · "
                    f"{dv['ucf_normal_alarms_per_min']} | {t['usrt']['verified_event_recall']} | "
                    f"{t['usrt']['false_alarms_per_min_outside_gt']} | {t['vtest_alarms_per_min']} | "
                    f"{t['mendeley']['event_recall']} · {t['mendeley']['fa_per_min']} | {t['ucf']['event_recall']} · {t['ucf']['fa_per_min']} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", action="append", required=True, help="name:usrt_replay:mendeley_replay:ucf_replay")
    ap.add_argument("--zoom-modes", default="hands", help="zoom modes the zoom replays were recorded with")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = list_profiles()
    results, inputs = {}, {}
    for spec in args.profile:
        name, usrt, men, ucf = spec.split(":")
        inputs[name] = {"usrt": usrt, "mendeley": men, "ucf": ucf}
        sets = sets_for(Path(usrt), Path(men), Path(ucf))
        results[name] = evaluate(sets, profile_params(profiles[name]), tuple(m for m in args.zoom_modes.split("+") if m))
        print(name, json.dumps(results[name]["test"]["usrt"]), flush=True)
    (out / "results.json").write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "inputs": inputs,
                                                  "profiles": {n: profiles[n] for n in results}, "results": results},
                                                 indent=2, default=float))
    (out / "table.md").write_text(table(results) + "\n")
    print(table(results))


if __name__ == "__main__":
    main()

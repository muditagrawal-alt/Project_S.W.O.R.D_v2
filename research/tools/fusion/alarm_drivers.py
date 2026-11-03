"""
Phase 5 error analysis: which evidence channel puts a person at >= SUSPICIOUS_ACTIVITY?

The rules engine (fusion/rules.py) is replayed over recordings with a probe that records, for every track-frame at a
candidate level >= SUSPICIOUS_ACTIVITY, the conditions that hold: weapon evidence >= w_susp, the learned weapon-action
head >= l_susp, PoseC3D violent / weapon groups >= a_susp, draw motion or aiming with weapon evidence >= w_low.
Run on the DEV sets of research/tools/fusion/tune_rules.py (mock-attack Cam1+Cam5, Mendeley P2 no-gun, UCF-Crime
dev-half normal videos) with the default rules and, optionally, a tuned operating point.

    python research/tools/fusion/alarm_drivers.py --usrt <replay> --mendeley <replay> --ucf <replay> --out <dir> \\
        [--tuning <tune_rules results.json> --operating-point 1.0] [--zoom-modes hands]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/fusion"))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/integration"))

from fusion.rules import FusionConfig, TemporalThreatEngine  # noqa: E402
from tune_rules import FILTERS_OFF, apply_det, cfg_for  # noqa: E402

from evaluate_profiles import sets_for  # noqa: E402


def drivers(e: dict, c: FusionConfig) -> list:
    w = TemporalThreatEngine._weapon(e)
    out = [name for name, cond in (
        ("weapon >= w_susp", w >= c.w_susp), ("learned head >= l_susp", e["act_weapon_learned"] >= c.l_susp),
        ("PoseC3D violent >= a_susp", e["act_violent"] >= c.a_susp), ("PoseC3D weapon >= a_susp", e["act_weapon"] >= c.a_susp),
        ("draw motion + weapon >= w_low", e["draw_motion"] >= c.draw_thr and w >= c.w_low),
        ("aiming + weapon >= w_low", e["aiming"] >= c.aim_thr and w >= c.w_low)) if cond]
    return out or ["other"]


class Probe(TemporalThreatEngine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.hits, self.n = Counter(), 0

    def _candidate(self, st, t):
        cand = super()._candidate(st, t)
        if cand >= 2:
            self.hits.update(drivers(st.ema, self.cfg))
            self.n += 1
        return cand


def probe(docs: list, params: dict) -> dict:
    hits, n, frames = Counter(), 0, 0
    for d in docs:
        eng = Probe(cfg_for(FusionConfig(), params, d["fps"]))
        latest = {}
        for f in d["frames"]:
            latest.update({int(k): v for k, v in (f.get("actions") or {}).items()})
            eng.update(f["timestamp"], f, (d["height"], d["width"]), latest)
        hits += eng.hits
        n += eng.n
        frames += len(d["frames"])
    return {"frames": frames, "track_frames_at_suspicious_or_above": n, "drivers": dict(hits.most_common())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usrt", required=True)
    ap.add_argument("--mendeley", required=True)
    ap.add_argument("--ucf", required=True)
    ap.add_argument("--tuning")
    ap.add_argument("--operating-point")
    ap.add_argument("--zoom-modes", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets = sets_for(Path(args.usrt), Path(args.mendeley), Path(args.ucf))
    base = FusionConfig()
    configs = {"default": {k: getattr(base, k) for k in ("w_susp", "w_event", "persist_s", "hold_s", "l_susp", "a_susp")}
               | {"zoom_min_score": 0.1, **FILTERS_OFF}}
    if args.tuning:
        tun = json.loads(Path(args.tuning).read_text())
        op = tun["operating_points"][args.operating_point] if args.operating_point else tun
        configs[f"tuned (budget {args.operating_point})" if args.operating_point else "tuned"] = op["selected"]["params"]
    zoom_modes = tuple(m for m in args.zoom_modes.split("+") if m)
    result = {"created": datetime.now().isoformat(timespec="seconds"), "inputs": vars(args), "configs": configs, "results": {}}
    for name, params in configs.items():
        result["results"][name] = {s: probe(apply_det(sets[s], params, zoom_modes), params)
                                   for s in ("usrt_dev", "men_dev_nogun", "ucf_dev_normal")}
        print(name, json.dumps(result["results"][name], indent=1), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alarm_drivers.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

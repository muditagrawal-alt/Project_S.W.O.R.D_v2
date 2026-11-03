"""
Write cross-fitted weapon-action scores (`weapon_action_prob`) into the Phase 4 estimates of recorded runs, in place.

Which head scores which video (no score comes from a model trained on that video):
  Mendeley videos at training locations P1/P2/P3  -> the fold head trained on the OTHER two locations
  everything else (Mendeley P4/P5, UCF-Crime, mock-attack, vtest) -> the deployment head (trained on P1+P2+P3)

    python research/tools/phase4/apply_weapon_head.py --head-run research/runs/<id>_weapon-head --recording <run dir>...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from action_recognition.weapon_head import WeaponActionHead  # noqa: E402

TRAIN_LOCS = ("P1", "P2", "P3")


def head_for(video: str, heads: dict) -> tuple:
    parts = video.split("/")
    if parts[0] == "mendeley":
        loc = parts[2].split("_")[2]
        if loc in TRAIN_LOCS:
            return heads[loc], f"fold:{loc}"
    return heads["deploy"], "deploy"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head-run", required=True)
    ap.add_argument("--recording", nargs="+", required=True)
    args = ap.parse_args()
    hr = Path(args.head_run)
    heads = {loc: WeaponActionHead(hr / f"fold_head_{loc}.json") for loc in TRAIN_LOCS}
    heads["deploy"] = WeaponActionHead(hr / "posec3d_weapon_head_v1.json")
    for rec in map(Path, args.recording):
        files = sorted(list(rec.glob("videos/*/frames.json")) + list(rec.glob("segments/*/frames.json")) + list(rec.glob("frames.json")))
        counts = {}
        for fj in files:
            doc = json.loads(fj.read_text())
            head, tag = head_for(doc["video"], heads)
            n = 0
            for f in doc["frames"]:
                for a in (f.get("actions") or {}).values():
                    if a.get("probabilities_120"):
                        a["weapon_action_prob"] = round(head.prob(a["probabilities_120"]), 4)
                        a["weapon_head"] = tag
                        n += 1
            doc.setdefault("sources", {})["weapon_action_head"] = f"{hr.name} (cross-fitted)"
            fj.write_text(json.dumps(doc))
            counts[tag] = counts.get(tag, 0) + n
        (rec / "weapon_head_applied.json").write_text(json.dumps({"head_run": str(hr), "estimates_by_head": counts}, indent=2))
        print(f"{rec}: {len(files)} recordings, estimates scored by head: {counts}")


if __name__ == "__main__":
    main()

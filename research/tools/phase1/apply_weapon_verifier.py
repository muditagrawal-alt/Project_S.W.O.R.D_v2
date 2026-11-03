"""
Write the second-stage verifier's p(weapon) ("verify") into every weapon detection of recorded runs, in place
(an added field; scores and boxes are unchanged). tracking.pipeline.rederive_weapons(verify_mode=..., verify_min=...)
then fuses it exactly as the live pipeline does, so dev tuning and evaluation can use the verifier without
re-running the detector.

    python research/tools/phase1/apply_weapon_verifier.py --weights weights/weapon_verifier_v1.pt --recording <run>...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "research/tools/integration"))

from detector_replay import source_frames  # noqa: E402
from tracking.verifier import WeaponVerifier  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(PROJECT_ROOT / "weights" / "weapon_verifier_v1.pt"))
    ap.add_argument("--recording", nargs="+", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    ver = WeaponVerifier(args.weights, device=args.device)
    for rec in map(Path, args.recording):
        files = sorted(list(rec.glob("videos/*/frames.json")) + list(rec.glob("segments/*/frames.json")))
        t0, n_det = time.time(), 0
        for fj in files:
            doc = json.loads(fj.read_text())
            by_idx = {f["frame_idx"]: f for f in doc["frames"]}
            for idx, img in source_frames(doc):
                f = by_idx.get(idx)
                if f is None or img is None or not f["weapon_detections"]:
                    continue
                p = ver.predict(img, [d["bbox"] for d in f["weapon_detections"]])
                for d, q in zip(f["weapon_detections"], p):
                    d["verify"] = round(float(q), 4)
                n_det += len(p)
            doc.setdefault("sources", {})["weapon_verifier"] = Path(args.weights).name
            fj.write_text(json.dumps(doc))
        (rec / "weapon_verifier_applied.json").write_text(json.dumps(
            {"weights": args.weights, "verifier_meta": ver.meta, "recordings": len(files), "detections_scored": n_det,
             "minutes": round((time.time() - t0) / 60, 1), "created": datetime.now().isoformat(timespec="seconds")}, indent=2))
        print(f"{rec}: {len(files)} recordings, {n_det} detections scored in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

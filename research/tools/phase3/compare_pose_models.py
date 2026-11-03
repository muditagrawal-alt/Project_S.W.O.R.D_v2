"""
Phase 3: does a bigger pose model link weapons to hands more often on real CCTV?

For every mock-attack frame, the recorded tracked person boxes are re-posed with each RTMPose tier (rtmlib:
lightweight = RTMPose-s, balanced = RTMPose-m (current), performance = RTMPose-x 384x288). Each ground-truth weapon
box is then associated with the pipeline's own rule (tracking/association.py). Metrics per tier: share of weapons
linked to a WRIST (what Phase 5's hand-held evidence needs), share linked at all, wrist keypoint confidence of the
linked person, and pose latency per person. Selection on DEV (Cam1+Cam5), TEST (Cam7) reported.

    python research/tools/phase3/compare_pose_models.py --recording research/runs/<full-pipeline run> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.association import associate  # noqa: E402

SEQ = PROJECT_ROOT / "research/data/mock_attack/sequences"


def main():
    import rtmlib

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--modes", nargs="+", default=["balanced", "lightweight", "performance"])
    ap.add_argument("--device", default="mps", help="rtmlib device ('mps' = CoreML on Apple silicon)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = []   # (camera, image path, persons, gt boxes)
    for fj in sorted(Path(args.recording).glob("segments/Cam*/frames.json")):
        rec = json.loads(fj.read_text())
        seq = json.loads((SEQ / rec["video"]).read_text())
        for f, g in zip(rec["frames"], seq["frames"]):
            if g["boxes"] and f["persons"]:
                work.append((rec["video"].split("-")[0], g["image"], f["persons"], [b["bbox"] for b in g["boxes"]]))
    print(f"{len(work)} frames with weapons and tracked persons", flush=True)
    results = {}
    for mode in args.modes:
        cfg = rtmlib.Body.MODE[mode]
        pose = rtmlib.RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"], backend="onnxruntime", device=args.device)
        stats = defaultdict(lambda: defaultdict(float))
        t_pose, n_people = 0.0, 0
        for cam, img_path, persons, gts in work:
            img = cv2.imread(img_path)
            t0 = time.perf_counter()
            kps, scs = pose(img, bboxes=[p["bbox"] for p in persons])
            t_pose += time.perf_counter() - t0
            n_people += len(persons)
            ps = [{"track_id": p["track_id"], "bbox": p["bbox"], "keypoints": np.asarray(k).tolist(), "scores": np.asarray(s).tolist()}
                  for p, k, s in zip(persons, kps, scs)]
            split = "dev" if cam in ("Cam1", "Cam5") else "test"
            for link in associate([{"bbox": b} for b in gts], ps):
                st = stats[split]
                st["weapons"] += 1
                st["linked"] += link["person_id"] is not None
                st["wrist"] += link["method"] == "wrist"
                if link["person_id"] is not None:
                    p = next(q for q in ps if q["track_id"] == link["person_id"])
                    st["wrist_conf_sum"] += max(p["scores"][9], p["scores"][10])
        results[mode] = {"model": cfg["pose"].split("/")[-1], "input": cfg["pose_input_size"],
                         "ms_per_person": round(1000 * t_pose / max(n_people, 1), 2)}
        for split, st in stats.items():
            results[mode][split] = {"weapons": int(st["weapons"]), "wrist_link_rate": round(st["wrist"] / max(st["weapons"], 1), 3),
                                    "any_link_rate": round(st["linked"] / max(st["weapons"], 1), 3),
                                    "mean_wrist_conf_linked": round(st["wrist_conf_sum"] / max(st["linked"], 1), 3)}
        print(mode, json.dumps(results[mode]), flush=True)
        best = max(results, key=lambda m: results[m].get("dev", {}).get("wrist_link_rate", 0))
        (out / "results.json").write_text(json.dumps({"created": datetime.now().isoformat(timespec="seconds"), "frames": len(work),
                                                      "selected_on_dev": best, "modes_done": list(results), "results": results},
                                                     indent=2))   # after every tier: a failed model download keeps the rest
    print("DEV-selected:", best)


if __name__ == "__main__":
    main()

"""
Conformance check for a Phase 2 module (see tracking/interface.py and tracking/PHASE2_INTERFACE.md).

    python -m tracking.contract --tracker mypackage.mymodule:MyTracker --video research/data/videos/vtest.avi
    python -m tracking.contract --tracker mypackage.mymodule:make_tracker --kwargs '{"fps": 10}' --max-frames 100

`--tracker` names a class or factory; it is called with --kwargs (JSON). The check feeds real frames and reports
every contract violation plus basic tracking statistics (IDs, track lengths, fragmentation, latency).
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import defaultdict
from typing import List

import numpy as np

TRAJECTORY_KEYS = {"speed_heights_s", "zone", "velocity_px_s", "acceleration_px_s2", "heading_deg", "dwell_s", "speed_m_s"}


def check_output(people, events, frame_shape, t) -> List[str]:
    errs = []
    h, w = frame_shape[:2]
    if not isinstance(people, list):
        return [f"t={t:.2f}: people must be a list, got {type(people).__name__}"]
    ids = []
    for p in people:
        if not isinstance(p, dict):
            errs.append(f"t={t:.2f}: each person must be a dict")
            continue
        for k in ("track_id", "bbox", "score"):
            if k not in p:
                errs.append(f"t={t:.2f}: person missing '{k}'")
        tid = p.get("track_id")
        if not isinstance(tid, (int, np.integer)):
            errs.append(f"t={t:.2f}: track_id must be int, got {type(tid).__name__}")
        ids.append(tid)
        b = p.get("bbox")
        if not (isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4):
            errs.append(f"t={t:.2f}: bbox must be [x1, y1, x2, y2]")
        else:
            x1, y1, x2, y2 = map(float, b)
            if not (x2 > x1 and y2 > y1):
                errs.append(f"t={t:.2f}: track {tid} bbox has non-positive size {b}")
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
                errs.append(f"t={t:.2f}: track {tid} bbox entirely outside the frame {b}")
        tr = p.get("trajectory")
        if tr is not None:
            if not isinstance(tr, dict):
                errs.append(f"t={t:.2f}: trajectory must be a dict or None")
            elif "speed_heights_s" in tr and tr["speed_heights_s"] is not None and not np.isfinite(float(tr["speed_heights_s"])):
                errs.append(f"t={t:.2f}: track {tid} speed_heights_s is not finite")
    if len(ids) != len(set(map(str, ids))):
        errs.append(f"t={t:.2f}: duplicate track_id in one frame {ids}")
    if not isinstance(events, list):
        errs.append(f"t={t:.2f}: zone_events must be a list")
    else:
        for e in events:
            if not (isinstance(e, dict) and e.get("type") in ("zone_enter", "zone_exit") and "track_id" in e and "zone" in e):
                errs.append(f"t={t:.2f}: bad zone event {e}")
    return errs


def run_check(tracker, video: str, max_frames: int = 300) -> dict:
    import cv2

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker.reset()
    errors, seen, lat, n = [], defaultdict(list), [], 0
    traj_keys = set()
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        t0 = time.perf_counter()
        people, events = tracker.update(frame, t)
        lat.append((time.perf_counter() - t0) * 1e3)
        errors += check_output(people, events, frame.shape, t)
        for p in people if isinstance(people, list) else []:
            if isinstance(p, dict) and "track_id" in p:
                seen[int(p["track_id"])].append(n)
                traj_keys |= set((p.get("trajectory") or {}).keys())
        n += 1
    cap.release()
    lengths = [len(v) for v in seen.values()]
    gaps = sum(sum(1 for a, b in zip(v, v[1:]) if b - a > 1) for v in seen.values())
    return {"frames": n, "violations": len(errors), "first_violations": errors[:20], "tracks": len(seen),
            "track_length_median": float(np.median(lengths)) if lengths else 0.0,
            "single_frame_tracks": int(sum(l == 1 for l in lengths)), "tracks_with_gaps": gaps,
            "trajectory_keys": sorted(traj_keys), "unknown_trajectory_keys": sorted(traj_keys - TRAJECTORY_KEYS),
            "latency_ms_mean": round(float(np.mean(lat)), 2) if lat else None,
            "reset_ok": hasattr(tracker, "reset"), "passed": not errors and n > 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", default="tracking.interface:StandInPhase2", help="module.path:ClassOrFactory")
    ap.add_argument("--kwargs", default='{"fps": 10, "device": "cpu"}')
    ap.add_argument("--video", default="research/data/videos/vtest.avi")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args()
    mod, attr = args.tracker.split(":")
    tracker = getattr(importlib.import_module(mod), attr)(**json.loads(args.kwargs))
    print(json.dumps(run_check(tracker, args.video, args.max_frames), indent=2))


if __name__ == "__main__":
    main()

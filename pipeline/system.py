"""
S.W.O.R.D. end-to-end system: Phase 1 -> Phase 2 (pluggable) -> Phase 3 -> Phase 4 -> Phase 5, frame by frame.

    sys = SwordSystem(fps=25, device="mps")
    for i, frame in enumerate(frames):
        rec = sys.process(frame, i, i / 25)      # sword.frames.v1 frame + actions + threat state + fusion events

Phase 2 is a slot: pass `phase2=` any object satisfying tracking.interface.Phase2Tracker (the team's module); the
default is a temporary stand-in. Phase 6 (service/) consumes the records and fusion events produced here.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

import numpy as np

from fusion.rules import FusionConfig, TemporalThreatEngine
from tracking.pipeline import DEFAULT_WEAPON_MODEL, SwordPipeline, pick_device


def _compact_action(r: dict) -> dict:
    keep = ("t", "window_start", "window_end", "n_frames", "partner_track_id", "top1_id", "top1_name", "top1_prob",
            "assigned_group", "threat_score", "group_probs", "top5", "heuristic_aiming_posture", "heuristic_aiming_conf",
            "probabilities_120", "latency_s")
    return {k: r[k] for k in keep if k in r}


class SwordSystem:
    def __init__(self, fps: float, device: Optional[str] = None, weapon_model=DEFAULT_WEAPON_MODEL,
                 use_ensemble: bool = False, use_pose: bool = True, use_action: bool = True,
                 fusion_cfg: Optional[FusionConfig] = None, phase2=None, zones=None, homography=None,
                 action_device: Optional[str] = None, pose_device: Optional[str] = None, async_action: bool = False,
                 weapon_imgsz: int = 640):
        import onnxruntime

        self.device = pick_device(device)
        if pose_device is None:  # rtmlib maps "mps" to onnxruntime's CoreML provider (2.5x faster on Apple silicon)
            pose_device = "mps" if "CoreMLExecutionProvider" in onnxruntime.get_available_providers() else "cpu"
        self.fps = fps
        self.p123 = SwordPipeline(fps, self.device, weapon_model=weapon_model, use_ensemble=use_ensemble,
                                  use_pose=use_pose, zones=zones, homography=homography, phase2=phase2,
                                  pose_device=pose_device, weapon_imgsz=weapon_imgsz)
        self.p4 = None
        if use_action and self.p123.pose is not None:
            from action_recognition.online import AsyncActionRecognizer, OnlineActionRecognizer

            # async (live): PoseC3D on a worker thread; sync (default): deterministic, for research runs
            self.p4 = (AsyncActionRecognizer if async_action else OnlineActionRecognizer)(device=action_device or self.device)
        self.p5 = TemporalThreatEngine(fusion_cfg or FusionConfig.for_fps(fps))
        self.timings = defaultdict(list)
        self.sources = {**self.p123.sources,
                        "phase4": (f"PoseC3D SlowOnly-R50 NTU-120 (online, 3 s windows / 1 s stride, linear temporal resampling, "
                                   f"{'async' if async_action else 'sync'})") if self.p4 else None,
                        "phase5": "TemporalThreatEngine (rules)"}

    def reset(self) -> None:
        self.p123.reset()
        if self.p4 is not None:
            self.p4.reset()
        self.p5.reset()
        self.timings.clear()

    def close(self) -> None:
        if self.p4 is not None and hasattr(self.p4, "close"):
            self.p4.close()

    def process(self, frame: np.ndarray, frame_idx: int, t: float) -> dict:
        t0 = time.perf_counter()
        rec = self.p123.step(frame, frame_idx, t)
        for k, v in self.p123.last_timing.items():
            self.timings[k].append(v)
        t1 = time.perf_counter()
        new = self.p4.update(t, rec["persons"], frame.shape) if self.p4 is not None else {}
        t2 = time.perf_counter()
        rec["actions"] = {int(k): _compact_action(v) for k, v in new.items()}
        latest = self.p4.latest if self.p4 is not None else {}
        for p in rec["persons"]:
            a = latest.get(p["track_id"])
            p["action"] = None if a is None else {k: a[k] for k in ("t", "top1_name", "top1_prob", "assigned_group",
                                                                    "threat_score") if k in a}
        state, events = self.p5.update(t, rec, frame.shape, latest)
        t3 = time.perf_counter()
        rec["threat"] = state
        rec["fusion_events"] = events
        self.timings["phase4_action_ms"].append((t2 - t1) * 1e3)
        self.timings["phase5_fusion_ms"].append((t3 - t2) * 1e3)
        self.timings["total_ms"].append((t3 - t0) * 1e3)
        return rec

    def timing_summary(self) -> dict:
        out = {}
        for k, v in self.timings.items():
            a = np.asarray(v)
            out[k] = {"mean": round(float(a.mean()), 2), "p50": round(float(np.median(a)), 2),
                      "p95": round(float(np.percentile(a, 95)), 2)} if len(a) else None
        if self.timings.get("total_ms"):
            out["throughput_fps"] = round(1000.0 / float(np.mean(self.timings["total_ms"])), 2)
        return out

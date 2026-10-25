"""
Phase 5 evaluation: baseline ladder for the research hypothesis + event-level metrics.

Hypothesis (roadmap): temporal behavioural/contextual information gives earlier and/or more reliable warning of
weapon-related events than frame-level weapon detection alone. Every rung below turns the SAME recorded pipeline
output into a per-frame alarm signal, so differences are attributable to the added information only:

    B0  frame        any weapon detection >= conf in the frame (frame-level detection alone)
    B0+ persist      B0 true in >= frac of the last window_s seconds (temporal smoothing only)
    B1  track        a weapon linked to a tracked person, persisting >= persist_s (Phase 2 stand-in + association)
    B1+ plausible    B1 with the CCTV plausibility prior on weapon size
    B2  hand         B1+ but only weapons linked to a wrist (Phase 3 pose)
    B3  rules-noact  temporal rules engine without PoseC3D (Phases 1-3)
    B4  rules-full   temporal rules engine with everything (Phases 1-4)

Metrics: alarm segments (rising edges, merged across gaps < merge_gap_s); false alarms per minute; time in alarm;
with ground-truth weapon segments: event recall, detection delay (negative = warning before the weapon was annotated
visible), false alarms per minute outside ground truth, frame-level precision/recall.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fusion.features import FeatureConfig, weapon_plausibility
from fusion.rules import LEVELS, FusionConfig, TemporalThreatEngine


# ───────────────────────────── segments & metrics ─────────────────────────────
def segments(times: Sequence[float], flags: Sequence[bool], merge_gap_s: float = 1.0) -> List[Tuple[float, float]]:
    segs: List[List[float]] = []
    for t, f in zip(times, flags):
        if not f:
            continue
        if segs and t - segs[-1][1] <= merge_gap_s:
            segs[-1][1] = t
        else:
            segs.append([t, t])
    return [(a, b) for a, b in segs]


def _event_windows(gt_segments, early_s: float, tol_s: float) -> List[Tuple[float, float, float, bool]]:
    """(gs, ge, window_start, isolated) per event. Early-warning credit needs a clean look-back: if the previous event
    ended less than early_s + tol_s before this one, the window starts at the event itself (no early credit)."""
    out = []
    for k, (gs, ge) in enumerate(gt_segments):
        isolated = k == 0 or gs - gt_segments[k - 1][1] >= early_s + tol_s
        out.append((gs, ge, gs - early_s if isolated else gs, isolated))
    return out


def _event_delay(times, flags, gs: float, ge: float, win_start: float, tol_s: float) -> Optional[float]:
    """Delay of the first alarm for one event (negative = warning before the weapon was annotated visible).
    An alarm already running when the window opens gets no early credit: delay 0 if still on at gs."""
    idx = [i for i, t in enumerate(times) if win_start - 1e-9 <= t <= ge + tol_s]
    if not idx:
        return None
    j = 0
    if idx[0] > 0 and flags[idx[0] - 1] and flags[idx[0]]:
        while j < len(idx) and flags[idx[j]]:
            j += 1
        if times[idx[j - 1]] >= gs - 1e-9:
            return 0.0
    for i in idx[j:]:
        if flags[i]:
            return float(times[i] - gs)
    return None


def evaluate_alarms(times: Sequence[float], flags: Sequence[bool], gt_segments: Optional[List[Tuple[float, float]]] = None,
                    gt_flags: Optional[Sequence[bool]] = None, tol_s: float = 1.0, early_s: float = 5.0,
                    merge_gap_s: float = 1.0) -> dict:
    times = np.asarray(times, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    duration_min = max((times[-1] - times[0]) / 60.0, 1e-9) if len(times) > 1 else 1e-9
    alarms = segments(times, flags, merge_gap_s)
    out = {"duration_min": round(duration_min, 3), "alarm_segments": len(alarms),
           "alarms_per_min": round(len(alarms) / duration_min, 3),
           "time_in_alarm_pct": round(100.0 * flags.mean(), 2) if len(flags) else 0.0}
    if gt_segments is None:
        return out
    detected, delays, iso_delays = 0, [], []
    for gs, ge, win_start, isolated in _event_windows(gt_segments, early_s, tol_s):
        d = _event_delay(times, flags, gs, ge, win_start, tol_s)
        if d is not None:
            detected += 1
            delays.append(d)
            if isolated:
                iso_delays.append(d)
    false = [a for a in alarms if not any(a[1] >= gs - tol_s and a[0] <= ge + tol_s for gs, ge in gt_segments)]
    gt_time = sum(ge - gs for gs, ge in gt_segments) / 60.0
    neg_min = max(duration_min - gt_time, 1e-9)
    out.update(gt_events=len(gt_segments), events_detected=detected,
               event_recall=round(detected / len(gt_segments), 3) if gt_segments else None,
               detection_delay_s_median=round(float(np.median(delays)), 3) if delays else None,
               detection_delay_s_mean=round(float(np.mean(delays)), 3) if delays else None,
               false_alarm_segments=len(false), false_alarms_per_min_outside_gt=round(len(false) / neg_min, 3),
               negative_min=round(neg_min, 4), delays_s=[round(float(d), 3) for d in delays],  # raw counts for pooling
               isolated_events=sum(1 for w in _event_windows(gt_segments, early_s, tol_s) if w[3]),
               isolated_delays_s=[round(float(d), 3) for d in iso_delays])
    if gt_flags is not None:
        g = np.asarray(gt_flags, dtype=bool)
        tp = int((flags & g).sum())
        out.update(frame_precision=round(tp / max(int(flags.sum()), 1), 3), frame_recall=round(tp / max(int(g.sum()), 1), 3),
                   frame_tp=tp, frame_alarms=int(flags.sum()), frame_gt=int(g.sum()))
    return out


# ───────────────────────────── baseline ladder ─────────────────────────────
def _times(frames: List[dict]) -> np.ndarray:
    return np.array([f["timestamp"] for f in frames], dtype=float)


def b0_frame(frames: List[dict], conf: float = 0.5) -> List[bool]:
    # raw per-frame detector output; weapon tracks would already carry ByteTrack's temporal filtering
    return [any(w["score"] >= conf for w in f.get("weapon_detections", f["weapons"])) for f in frames]


def b0_persist(frames: List[dict], conf: float = 0.5, window_s: float = 1.0, frac: float = 0.6) -> List[bool]:
    t = _times(frames)
    raw = np.array(b0_frame(frames, conf))
    out = []
    for i, ti in enumerate(t):
        m = (t >= ti - window_s) & (t <= ti)
        out.append(bool(raw[m].mean() >= frac))
    return out


def b1_track(frames: List[dict], frame_shape, conf: float = 0.5, persist_s: float = 1.0, grace_s: float = 0.6,
             plausible: bool = False, hand_only: bool = False, cfg: Optional[FeatureConfig] = None,
             return_ids: bool = False) -> list:
    cfg = cfg or FeatureConfig()
    area = float(frame_shape[0] * frame_shape[1])
    since: Dict[int, float] = {}
    last: Dict[int, float] = {}
    out = []
    for f in frames:
        t = f["timestamp"]
        persons = {p["track_id"]: p for p in f["persons"]}
        on = set()
        for w in f["weapons"]:
            pid = w["person_id"]
            if pid is None or (hand_only and w.get("assoc_method") != "wrist"):
                continue
            s = w["score"] * (weapon_plausibility(w, area, persons.get(pid), cfg) if plausible else 1.0)
            if s >= conf:
                on.add(pid)
        for pid in on:
            if pid not in since or t - last.get(pid, -1e9) > grace_s:
                since[pid] = t
            last[pid] = t
        ids = {pid for pid in since if t - since[pid] >= persist_s and t - last[pid] <= grace_s}
        out.append(ids if return_ids else bool(ids))
    return out


def replay_rules(frames: List[dict], frame_shape, cfg: FusionConfig, with_tracks: bool = False) -> list:
    """Re-run the rules engine over a recorded run (frames must carry their Phase 4 'actions' updates).
    with_tracks: also return {track_id: level index} per frame (who caused the scene level)."""
    eng = TemporalThreatEngine(copy.deepcopy(cfg))
    latest: Dict[int, dict] = {}
    levels, tracks = [], []
    for f in frames:
        latest.update({int(k): v for k, v in (f.get("actions") or {}).items()})
        state, _ = eng.update(f["timestamp"], f, frame_shape, latest)
        levels.append(state["scene_level"])
        tracks.append({int(k): LEVELS.index(v["level"]) for k, v in state["tracks"].items()})
    return (levels, tracks) if with_tracks else levels


def ladder_details(frames: List[dict], frame_shape, conf: float = 0.5, cfg: Optional[FusionConfig] = None,
                   alarm_level: str = "SUSPICIOUS_ACTIVITY") -> Dict[str, dict]:
    """Every rung as {"flags": per-frame alarm, "ids": per-frame alarming person ids (person-based rungs) or None}."""
    cfg = cfg or FusionConfig()
    thr = LEVELS.index(alarm_level)
    no_act = copy.deepcopy(cfg)
    no_act.use_action = False
    pg = {"persist_s": cfg.persist_s, "grace_s": cfg.grace_s}  # same persistence rule as the engine
    out = {"B0_frame": {"flags": b0_frame(frames, conf), "ids": None},
           "B0+_persist": {"flags": b0_persist(frames, conf), "ids": None}}
    for name, kw in (("B1_track", {}), ("B1+_plausible", {"plausible": True}),
                     ("B2_hand", {"plausible": True, "hand_only": True})):
        ids = b1_track(frames, frame_shape, conf, cfg=cfg.features, return_ids=True, **kw, **pg)
        out[name] = {"flags": [bool(x) for x in ids], "ids": ids}
    lv_no, tr_no = replay_rules(frames, frame_shape, no_act, with_tracks=True)
    lv, tr = replay_rules(frames, frame_shape, cfg, with_tracks=True)
    out["B3_rules_noaction"] = {"flags": [LEVELS.index(x) >= thr for x in lv_no],
                                "ids": [{k for k, v in t.items() if v >= thr} for t in tr_no]}
    out["B4_rules_full"] = {"flags": [LEVELS.index(x) >= thr for x in lv], "ids": [{k for k, v in t.items() if v >= thr} for t in tr]}
    out["B4_rules_full_WRE_only"] = {"flags": [LEVELS.index(x) >= 3 for x in lv], "ids": [{k for k, v in t.items() if v >= 3} for t in tr]}
    return out


def baseline_ladder(frames: List[dict], frame_shape, conf: float = 0.5, cfg: Optional[FusionConfig] = None,
                    alarm_level: str = "SUSPICIOUS_ACTIVITY") -> Dict[str, List[bool]]:
    return {k: v["flags"] for k, v in ladder_details(frames, frame_shape, conf, cfg, alarm_level).items()}


# ───────────────────────────── spatially verified metrics ─────────────────────────────
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def verified_metrics(frames: List[dict], details: Dict[str, dict], gt_boxes: List[list], gt_segments, conf: float = 0.5,
                     iou_thr: float = 0.1, early_s: float = 5.0, tol_s: float = 1.0, merge_gap_s: float = 1.0) -> Dict[str, dict]:
    """Event metrics that require the alarm to point at the weapon, not merely coincide with it in time.

    Detection rungs (B0, B0+): a frame counts if a detection >= conf overlaps a ground-truth weapon (IoU >= iou_thr,
    lenient because CCTV weapons are tiny). Person rungs (B1..B4): it counts if an alarmed person is a carrier of
    that ground-truth event — a person the pipeline's own association links to a ground-truth weapon box during
    the event. Alarm segments that overlap an event in time but never point at it are 'misattributed'.
    """
    from tracking.association import associate

    times = [f["timestamp"] for f in frames]
    det_hit = [any(d["score"] >= conf and any(_iou(d["bbox"], g) >= iou_thr for g in gt_boxes[i])
                   for d in f.get("weapon_detections", f["weapons"])) for i, f in enumerate(frames)]
    carriers_by_frame = []
    for i, f in enumerate(frames):
        links = associate([{"bbox": g} for g in gt_boxes[i]], f["persons"]) if gt_boxes[i] else []
        carriers_by_frame.append({a["person_id"] for a in links if a["person_id"] is not None})
    t_arr = np.asarray(times, dtype=float)
    hit_t = t_arr[np.asarray(det_hit, dtype=bool)] if len(t_arr) else t_arr

    def det_window(win: float) -> List[bool]:   # a correctly placed detection within [t - win, t]
        return [bool(((hit_t >= t - win - 1e-9) & (hit_t <= t + 1e-9)).any()) for t in t_arr]

    det_win = {0.0: det_window(0.0), 1.0: det_window(1.0)}
    out = {}
    for name, d in details.items():
        flags, ids = d["flags"], d["ids"]
        detected, delays, iso_delays, verified_any = 0, [], [], []
        for gs, ge, win_start, isolated in _event_windows(gt_segments, early_s, tol_s):
            carriers = set().union(*[carriers_by_frame[i] for i, t in enumerate(times) if gs <= t <= ge] or [set()])
            if ids is None:   # detection rungs: a correctly placed detection in the same frame (B0) / B0+'s window
                win = 0.0 if name == "B0_frame" else 1.0
                dw = det_win[win]
                ok_flags = [bool(flags[i]) and dw[i] if win_start - 1e-9 <= times[i] <= ge + tol_s else False
                            for i in range(len(times))]
            else:
                ok_flags = [bool(flags[i]) and bool(ids[i] & carriers) if win_start - 1e-9 <= times[i] <= ge + tol_s else False
                            for i in range(len(times))]
            d = _event_delay(times, ok_flags, gs, ge, win_start, tol_s)
            verified_any.append(d is not None)
            if d is not None:
                detected += 1
                delays.append(d)
                if isolated:
                    iso_delays.append(d)
        alarms = segments(times, flags, merge_gap_s)
        overlapping = [a for a in alarms if any(a[1] >= gs - tol_s and a[0] <= ge + tol_s for gs, ge in gt_segments)]
        misattributed = 0
        for a0, a1 in overlapping:
            seg_idx = [k for k, (gs, ge) in enumerate(gt_segments) if a1 >= gs - tol_s and a0 <= ge + tol_s]
            if not any(verified_any[k] for k in seg_idx):
                misattributed += 1
        out[name] = {"verified_events_detected": detected,
                     "verified_event_recall": round(detected / len(gt_segments), 3) if gt_segments else None,
                     "verified_delay_s_median": round(float(np.median(delays)), 3) if delays else None,
                     "verified_delays_s": [round(float(x), 3) for x in delays],
                     "verified_isolated_delays_s": [round(float(x), 3) for x in iso_delays],
                     "misattributed_alarm_segments": misattributed}
    return out

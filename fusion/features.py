"""
Phase 5 inputs: per-person evidence channels built from the shared frame record (Phases 1-4).

Channels (all in [0, 1] unless noted):
    weapon_hand     weapon linked to one of this person's wrists (Phase 1 x Phase 3), x plausibility
    weapon_body     weapon box mostly inside this person's box (no pose link), x plausibility
    weapon_near     unlinked weapon within ~1 body height of the person, x plausibility
    act_weapon      PoseC3D WEAPON_THREAT group probability (Phase 4), 0 if the estimate is stale
    act_violent     PoseC3D VIOLENT_ATTACK group probability
    act_suspicious  PoseC3D SUSPICIOUS_AGGRESSION group probability
    aiming          aiming-posture heuristic fired (0/1) — unvalidated, low weight
    draw_motion     hand near waist AND fast wrist (torso lengths/s), squashed to [0, 1]
    running         body-height-normalised speed above a walking pace, squashed to [0, 1]
    restricted      person's footpoint inside a zone marked restricted (0/1)

Plausibility prior (motivated by the vtest false-alarm finding, where a curved road was detected as a frame-filling
"Sword"): a hand-held weapon in surveillance footage is small relative to the frame and to the person holding it.
Boxes covering > max_frame_area of the frame, or > max_person_area of the linked person's box, are discounted to 0
(linear ramp over the following 50%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

CHANNELS = ["weapon_hand", "weapon_body", "weapon_near", "act_weapon", "act_violent", "act_suspicious",
            "aiming", "draw_motion", "running", "restricted"]


@dataclass
class FeatureConfig:
    max_frame_area: float = 0.15       # weapon box / frame area
    max_person_area: float = 0.6       # weapon box / linked person box area
    near_dist_heights: float = 1.0     # "near" = weapon centre within this many person heights
    action_max_age_s: float = 2.5      # PoseC3D estimates older than this are ignored
    walk_speed_heights_s: float = 0.9  # ~1.5 m/s for a 1.7 m person
    draw_speed_norm: float = 2.0       # wrist speed (torso lengths/s) that counts as a fast movement
    restricted_zones: List[str] = field(default_factory=list)


def _ramp(x: float, lo: float) -> float:
    """1 below lo, linear to 0 at 1.5 * lo."""
    if x <= lo:
        return 1.0
    return float(max(0.0, 1.0 - (x - lo) / (0.5 * lo)))


def weapon_plausibility(w: dict, frame_area: float, person: Optional[dict], cfg: FeatureConfig) -> float:
    x1, y1, x2, y2 = w["bbox"]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    p = _ramp(area / max(frame_area, 1.0), cfg.max_frame_area)
    if person is not None:
        px1, py1, px2, py2 = person["bbox"]
        p_area = max(1.0, (px2 - px1) * (py2 - py1))
        p *= _ramp(area / p_area, cfg.max_person_area)
    return p


def person_features(person: dict, weapons: List[dict], frame_shape, t: float, action: Optional[dict],
                    cfg: FeatureConfig) -> Dict[str, float]:
    h_img, w_img = frame_shape[:2]
    frame_area = float(h_img * w_img)
    tid = person["track_id"]
    f = dict.fromkeys(CHANNELS, 0.0)
    px1, py1, px2, py2 = person["bbox"]
    ph = max(1.0, py2 - py1)
    pc = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
    for w in weapons:
        if w["person_id"] == tid:
            s = w["score"] * weapon_plausibility(w, frame_area, person, cfg)
            key = "weapon_hand" if w.get("assoc_method") == "wrist" else "weapon_body"
            f[key] = max(f[key], s)
        elif w["person_id"] is None:
            wc = np.array([(w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2])
            if np.hypot(*(wc - pc)) / ph <= cfg.near_dist_heights:
                f["weapon_near"] = max(f["weapon_near"], w["score"] * weapon_plausibility(w, frame_area, None, cfg))
    if action is not None and t - action.get("t", -1e9) <= cfg.action_max_age_s:
        gp = action.get("group_probs", {})
        f["act_weapon"] = float(gp.get("prob_WEAPON_THREAT", 0.0))
        f["act_violent"] = float(gp.get("prob_VIOLENT_ATTACK", 0.0))
        f["act_suspicious"] = float(gp.get("prob_SUSPICIOUS_AGGRESSION", 0.0))
        f["aiming"] = float(bool(action.get("heuristic_aiming_posture", False)))
    kin = person.get("kinematics") or {}
    wv = kin.get("wrist_velocity") or {}
    speeds = [v for v in (wv.get("left_wrist_speed_norm"), wv.get("right_wrist_speed_norm")) if v is not None]
    if (kin.get("hand_to_waist") or {}).get("hands_near_waist") and speeds:
        f["draw_motion"] = float(min(1.0, max(speeds) / cfg.draw_speed_norm))
    tr = person.get("trajectory") or {}
    sp = tr.get("speed_heights_s")
    if sp is not None:
        f["running"] = float(np.clip((sp - cfg.walk_speed_heights_s) / cfg.walk_speed_heights_s, 0.0, 1.0))
    if tr.get("zone") in cfg.restricted_zones:
        f["restricted"] = 1.0
    return f


def scene_features(weapons: List[dict], frame_shape, cfg: FeatureConfig) -> Dict[str, float]:
    """Weapons not linked to anyone (on a table, dropped, or false alarms), plausibility-weighted."""
    h, w = frame_shape[:2]
    best = 0.0
    for wp in weapons:
        if wp["person_id"] is None:
            best = max(best, wp["score"] * weapon_plausibility(wp, float(h * w), None, cfg))
    return {"weapon_unattended": best}

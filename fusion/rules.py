"""
Phase 5 — temporal threat fusion, transparent rule-based engine.

Why rules first: the roadmap's Temporal Transformer needs videos labelled with NORMAL / ELEVATED / SUSPICIOUS ACTIVITY
/ WEAPON-RELATED EVENT intervals, which do not exist yet. This engine fuses the same inputs with explicit, hand-set
rules so the system works now, its decisions are explainable, and it serves as the baseline the transformer
(fusion/transformer.py) must beat. Levels describe observable events, never a person's identity or intent.

Per tracked person, every evidence channel (fusion/features.py) is smoothed with an exponential moving average
(time constant tau_s). Weapon evidence must also persist (continuously above weapon_on, with a short grace gap).
A candidate level is computed from the smoothed evidence; the person's level rises immediately to the candidate
and falls only after the candidate has stayed lower for hold_s (hysteresis). The scene level is the maximum over
people (a person who just vanished keeps their level for hold_s), plus an "unattended weapon" rule for plausible
weapons linked to nobody (with the same hysteresis).

Ablation switches (use_*) remove whole evidence sources so the contribution of each phase can be measured.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from fusion.features import CHANNELS, FeatureConfig, person_features, scene_features

LEVELS = ["NORMAL", "ELEVATED", "SUSPICIOUS_ACTIVITY", "WEAPON_RELATED_EVENT"]


@dataclass
class FusionConfig:
    tau_s: float = 0.7             # EMA time constant for all evidence channels
    weapon_on: float = 0.35        # raw weapon evidence counted as "present" for persistence
    grace_s: float = 0.6           # gap tolerated inside a persistent weapon observation
    persist_s: float = 1.0         # weapon must persist this long for WEAPON_RELATED_EVENT
    w_low: float = 0.25            # smoothed weapon evidence thresholds
    w_susp: float = 0.40
    w_event: float = 0.50
    # Action-group thresholds were set ABOVE the noise PoseC3D shows on weapon-free footage (development clip
    # vtest.avi, 376 clips: WEAPON_THREAT max 0.386 / p99 0.096; VIOLENT_ATTACK max 0.220; SUSPICIOUS_AGGRESSION
    # without 'follow other person' max 0.031). The mock-attack CCTV set is kept untouched for testing.
    a_low: float = 0.40            # smoothed action-group probability thresholds
    a_susp: float = 0.55
    a_event: float = 0.40
    a_elev_suspicious: float = 0.30
    draw_thr: float = 0.5
    aim_thr: float = 0.5
    run_thr: float = 0.5
    restricted_dwell_s: float = 2.0
    hold_s: float = 2.0            # minimum time below a level before stepping down
    unattended_persist_s: float = 2.0
    use_weapon: bool = True
    use_pose: bool = True          # wrist linking / draw motion / aiming (Phase 3)
    use_action: bool = True        # PoseC3D groups (Phase 4)
    use_trajectory: bool = True    # running / restricted zones (Phase 2 stand-in)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @classmethod
    def for_fps(cls, fps: float, **overrides) -> "FusionConfig":
        """Defaults adapted to the frame rate, fixed a priori (not tuned on any test data): the persistence grace
        must tolerate one missed detection, i.e. a gap of two frame intervals (1.0 s at 2 fps, 0.08 s at 25 fps)."""
        cfg = cls(**overrides)
        if "grace_s" not in overrides:
            cfg.grace_s = max(cfg.grace_s, 2.0 / fps + 1e-3)
        return cfg


class _TrackState:
    def __init__(self):
        self.ema = dict.fromkeys(CHANNELS, 0.0)
        self.t = None
        self.weapon_on_since: Optional[float] = None
        self.weapon_last_on: Optional[float] = None
        self.level = 0
        self.lower_since: Optional[float] = None
        self.restricted_since: Optional[float] = None


class TemporalThreatEngine:
    def __init__(self, cfg: Optional[FusionConfig] = None, stale_s: float = 5.0):
        self.cfg = cfg or FusionConfig()
        self.tracks: Dict[int, _TrackState] = {}
        self.scene_ema = 0.0
        self.scene_t: Optional[float] = None
        self.unattended_since: Optional[float] = None
        self.frame_t: Optional[float] = None   # previous frame's timestamp (observed frame interval)
        self.scene_level = 0
        self.unattended_level = 0
        self.unattended_lower_since: Optional[float] = None
        self.stale_s = stale_s

    def reset(self) -> None:
        self.__init__(self.cfg, self.stale_s)

    def _mask(self, f: Dict[str, float]) -> Dict[str, float]:
        c = self.cfg
        f = dict(f)
        if not c.use_weapon:
            for k in ("weapon_hand", "weapon_body", "weapon_near"):
                f[k] = 0.0
        if not c.use_pose:  # without pose a hand link cannot exist: fall back to body containment
            f["weapon_body"] = max(f["weapon_body"], f["weapon_hand"])
            f["weapon_hand"] = f["draw_motion"] = f["aiming"] = 0.0
        if not c.use_action:
            f["act_weapon"] = f["act_violent"] = f["act_suspicious"] = f["aiming"] = 0.0
        if not c.use_trajectory:
            f["running"] = f["restricted"] = 0.0
        return f

    @staticmethod
    def _weapon(f: Dict[str, float]) -> float:
        return max(f["weapon_hand"], 0.8 * f["weapon_body"], 0.5 * f["weapon_near"])

    def _candidate(self, st: _TrackState, t: float) -> int:
        c, e = self.cfg, st.ema
        w = self._weapon(e)
        persisted = st.weapon_on_since is not None and t - st.weapon_on_since >= c.persist_s
        if (w >= c.w_event and persisted) or (w >= c.w_susp and e["act_weapon"] >= c.a_event):
            return 3
        if (w >= c.w_susp or e["act_violent"] >= c.a_susp or e["act_weapon"] >= c.a_susp
                or (e["draw_motion"] >= c.draw_thr and w >= c.w_low) or (e["aiming"] >= c.aim_thr and w >= c.w_low)):
            return 2
        restricted = st.restricted_since is not None and t - st.restricted_since >= c.restricted_dwell_s
        if (w >= c.w_low or e["act_suspicious"] >= c.a_elev_suspicious or e["act_violent"] >= c.a_low
                or e["act_weapon"] >= c.a_low or e["running"] >= c.run_thr or restricted):
            return 1
        return 0

    @staticmethod
    def _step_level(level: int, cand: int, lower_since: Optional[float], t: float, hold_s: float):
        if cand >= level:
            return cand, None
        if lower_since is None:
            return level, t
        if t - lower_since >= hold_s:
            return cand, None
        return level, lower_since

    def update(self, t: float, frame: dict, frame_shape, actions: Dict[int, dict]) -> Tuple[dict, List[dict]]:
        """frame: one sword.frames.v1 frame (persons, weapons). actions: latest Phase 4 result per track id."""
        c = self.cfg
        events, out_tracks = [], {}
        # a new track's first observation counts as one frame interval of evidence (frame-rate independent)
        dt_new = (t - self.frame_t) if self.frame_t is not None and t > self.frame_t else 1.0 / 25.0
        self.frame_t = t
        for p in frame["persons"]:
            tid = p["track_id"]
            st = self.tracks.setdefault(tid, _TrackState())
            f = self._mask(person_features(p, frame["weapons"], frame_shape, t, actions.get(tid), c.features))
            # a new track starts from zero evidence (a single first observation must not jump straight to its value)
            dt = dt_new if st.t is None else max(t - st.t, 0.0)
            alpha = 1.0 - math.exp(-dt / c.tau_s)
            for k in CHANNELS:
                st.ema[k] += alpha * (f[k] - st.ema[k])
            st.t = t
            if self._weapon(f) >= c.weapon_on:
                if st.weapon_on_since is None or (st.weapon_last_on is not None and t - st.weapon_last_on > c.grace_s):
                    st.weapon_on_since = t
                st.weapon_last_on = t
            elif st.weapon_last_on is not None and t - st.weapon_last_on > c.grace_s:
                st.weapon_on_since = None
            st.restricted_since = (st.restricted_since or t) if f["restricted"] > 0 else None
            prev = st.level
            st.level, st.lower_since = self._step_level(st.level, self._candidate(st, t), st.lower_since, t, c.hold_s)
            if st.level != prev:
                events.append({"type": "level_change", "scope": "track", "track_id": tid, "t": round(t, 3),
                               "from": LEVELS[prev], "to": LEVELS[st.level],
                               "evidence": {k: round(v, 3) for k, v in st.ema.items() if v > 0.01}})
            out_tracks[tid] = {"level": LEVELS[st.level], "evidence": {k: round(v, 3) for k, v in st.ema.items()}}
        # unattended weapons (scene)
        sf = scene_features(frame["weapons"], frame_shape, c.features) if c.use_weapon else {"weapon_unattended": 0.0}
        alpha = 1.0 - math.exp(-(dt_new if self.scene_t is None else max(t - self.scene_t, 0.0)) / c.tau_s)
        self.scene_ema += alpha * (sf["weapon_unattended"] - self.scene_ema)
        self.scene_t = t
        self.unattended_since = (self.unattended_since or t) if self.scene_ema >= c.w_low else None
        unattended_cand = 1 if self.unattended_since is not None and t - self.unattended_since >= c.unattended_persist_s else 0
        self.unattended_level, self.unattended_lower_since = self._step_level(
            self.unattended_level, unattended_cand, self.unattended_lower_since, t, c.hold_s)
        # Track levels already carry hysteresis, so the scene takes their maximum directly (a second hold here would
        # double the decay time). A track that vanished (occlusion, tracker loss) keeps its level for hold_s.
        lost = [st.level for tid, st in self.tracks.items()
                if tid not in out_tracks and st.t is not None and t - st.t <= c.hold_s]
        prev = self.scene_level
        self.scene_level = max([self.unattended_level] + lost + [LEVELS.index(v["level"]) for v in out_tracks.values()])
        if self.scene_level != prev:
            events.append({"type": "level_change", "scope": "scene", "track_id": None, "t": round(t, 3),
                           "from": LEVELS[prev], "to": LEVELS[self.scene_level],
                           "tracks_at_level": [k for k, v in out_tracks.items() if LEVELS.index(v["level"]) == self.scene_level]})
        for tid in [k for k, st in self.tracks.items() if st.t is not None and t - st.t > self.stale_s]:
            del self.tracks[tid]
        state = {"scene_level": LEVELS[self.scene_level], "weapon_unattended_ema": round(self.scene_ema, 3), "tracks": out_tracks}
        return state, events

    def config_dict(self) -> dict:
        return asdict(self.cfg)

"""
Named system profiles: one place that says which detector, input size, zoom, thresholds and fusion settings a
configuration uses, so the CLI, the Phase 6 service and the command centre run exactly the evaluated setups.

Profiles live in config/profiles.json (written by research/tools/integration/install_profile.py from a dev-tuning
result); "original" is built in. Weights are referenced relative to the project root.

    from pipeline.profiles import build_system, list_profiles
    system = build_system("cctv-zoom", fps=10)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILES_FILE = PROJECT_ROOT / "config" / "profiles.json"

BUILTIN = {
    "original": {
        "description": "S.W.O.R.D. as shipped: YOLO26s trained on web images, full frame at 640 px, default rules",
        "weapon_model": "weights/yolo26s_best.pt", "weapon_imgsz": 640, "weapon_zoom": [], "zoom_min_score": 0.0,
        "fusion": {}, "evaluated_in": "research/runs/20261025-integration-summary",
    },
}


def list_profiles() -> Dict[str, dict]:
    profiles = dict(BUILTIN)
    if PROFILES_FILE.exists():
        profiles.update(json.loads(PROFILES_FILE.read_text()).get("profiles", {}))
    return profiles


def default_profile() -> str:
    if PROFILES_FILE.exists():
        return json.loads(PROFILES_FILE.read_text()).get("default", "original")
    return "original"


def system_kwargs(name: str) -> dict:
    p = list_profiles()[name]
    model = PROJECT_ROOT / p["weapon_model"]
    if not model.exists():
        raise FileNotFoundError(f"profile {name!r}: weights not found: {model}")
    return {"weapon_model": str(model), "weapon_imgsz": int(p.get("weapon_imgsz", 640)),
            "weapon_zoom": tuple(p.get("weapon_zoom", ())), "zoom_min_score": float(p.get("zoom_min_score", 0.0)),
            "zoom_min_window": int(p.get("zoom_min_window", 0)), "weapon_max_area": float(p.get("weapon_max_area", 1.0)),
            **({"weapon_tta_scales": tuple(int(v) for v in p["weapon_tta_scales"])} if p.get("weapon_tta_scales") else {}),
            **{k: True for k in ("weapon_tta_flip", "zoom_flip") if p.get(k)},
            **({"weapon_verifier": str(PROJECT_ROOT / p["weapon_verifier"]), "verify_mode": p.get("verify_mode", "geo"),
                "verify_min": float(p.get("verify_min", 0.0))} if p.get("weapon_verifier") else {})}


def fusion_config(name: str, fps: float):
    from fusion.rules import FusionConfig

    return FusionConfig.for_fps(fps, **list_profiles()[name].get("fusion", {}))


PHASE2_KINDS = ("standin", "botsort", "persistent", "persistent-legacy")


def make_phase2(kind: str, fps: float, device=None, track_seconds: float | None = None):
    """Phase 2 module by name: the stand-in (ByteTrack; None = SwordSystem's default instance), the stand-in with
    BoT-SORT, the team's persistent_video_tracking package (tracking/persistent_phase2.py), or that package as
    submitted (benchmarking only). track_seconds: how long a lost person is kept (stand-in and BoT-SORT)."""
    if kind not in PHASE2_KINDS:
        raise ValueError(f"unknown Phase 2 module {kind!r}; choose from {PHASE2_KINDS}")
    if kind == "standin" and track_seconds is None:
        return None
    from tracking.pipeline import pick_device

    device = pick_device(device)
    if kind in ("standin", "botsort"):
        from tracking.interface import BoTSORTPhase2, StandInPhase2

        return (StandInPhase2 if kind == "standin" else BoTSORTPhase2)(fps, device, track_seconds=track_seconds or 1.5)
    from tracking.persistent_phase2 import PersistentPhase2

    return PersistentPhase2(fps, device=device, legacy=kind == "persistent-legacy")


def build_system(name: str, fps: float, phase2: str | None = None, track_seconds: float | None = None, **overrides):
    """phase2 / track_seconds: Phase 2 module (PHASE2_KINDS) and lost-track time; default: the profile's "phase2" and
    "track_seconds" entries, else the stand-in with 1.5 s."""
    from pipeline.system import SwordSystem

    kw = system_kwargs(name) | overrides
    kw.setdefault("fusion_cfg", fusion_config(name, fps))
    prof = list_profiles()[name]
    module = make_phase2(phase2 or prof.get("phase2", "standin"), fps, kw.get("device"),
                         track_seconds if track_seconds is not None else prof.get("track_seconds"))
    if module is not None:
        kw["phase2"] = module
    return SwordSystem(fps=fps, **kw)

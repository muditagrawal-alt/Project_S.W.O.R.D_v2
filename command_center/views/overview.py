"""Overview: system status, models, the best current configuration and headline results."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import (PROJECT_ROOT, RUNS, ServiceClient, active_jobs, detector_registry,  # noqa: E402
                 list_runs, system_status, weapon_head_status)
from pipeline.profiles import default_profile, list_profiles  # noqa: E402

st.title("🛡️ S.W.O.R.D. command centre")
st.caption("Weapon detection → tracking → pose → action recognition → temporal threat fusion → incident intelligence. "
           "Levels describe observable events, never identity or intent.")

sysinfo = system_status()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Apple GPU (MPS)", "available" if sysinfo.get("mps") else "not available")
c2.metric("Pose acceleration (CoreML)", "on" if sysinfo.get("coreml") else "off")
batt = sysinfo.get("battery") or {}
c3.metric("Battery", f"{batt.get('percent', '?')}%", ("on charger" if batt.get("ac") else "on battery") if batt else None,
          delta_color="normal" if batt.get("ac") else "inverse")
c4.metric("Disk free", f"{sysinfo.get('disk_free_gb')} GB")
health = ServiceClient(st.session_state.get("service_url", "http://127.0.0.1:8000")).health()
c5.metric("Phase 6 service", "running" if health else "stopped")
if batt and not batt.get("ac") and batt.get("percent", 100) < 30:
    st.warning("Battery is low and the charger is not connected: macOS will sleep and long runs will pause.")

st.subheader("Phases")
head = weapon_head_status()
dp = default_profile()
DP = list_profiles()[dp]
p1 = (f"profile '{dp}': {Path(DP['weapon_model']).name} @ {DP.get('weapon_imgsz')} px"
      + (f" + {'+'.join(DP['weapon_zoom'])} zoom" if DP.get("weapon_zoom") else ""))
phases = pd.DataFrame([
    ["1 Weapon detection", p1, "CCTV fine-tuned YOLO26s; person-zoom crops; box-size and magnification limits"],
    ["2 Tracking & trajectories", "stand-in (YOLO26n + ByteTrack)", "slot for the team's Phase 2 module (tracking/PHASE2_INTERFACE.md)"],
    ["3 Pose", "RTMPose-m (rtmlib, CoreML)", "top-down on tracked people"],
    ["4 Action recognition", "PoseC3D NTU-120 + learned weapon-action head" if head.get("installed") else "PoseC3D NTU-120",
     "linear temporal resampling; head cross-fitted on Mendeley P1-P3"],
    ["5 Temporal fusion", "rules engine (default) · Temporal Transformer (research)", "smoothed evidence, persistence, hysteresis"],
    ["6 Incident intelligence", "FastAPI + WebSocket + SQLite/PostgreSQL", "evidence clips with SHA-256, audit log"],
], columns=["Phase", "Current component", "Notes"])
st.dataframe(phases, hide_index=True, width="stretch")

st.subheader("System profiles")
st.caption("A profile fixes the detector, input size, person-zoom, zoom threshold and fusion rules of one evaluated "
           "configuration; the CLI (--profile), the service and the command centre all run profiles by name.")
st.dataframe(pd.DataFrame([{"profile": f"{n} (default)" if n == dp else n, "detector": p["weapon_model"],
                            "input px": p.get("weapon_imgsz"), "zoom": "+".join(p.get("weapon_zoom", [])) or "off",
                            "zoom min score": p.get("zoom_min_score", 0.0), "zoom min window px": p.get("zoom_min_window", 0),
                            "max box % of frame": round(100 * p.get("weapon_max_area", 1.0), 1),
                            "fusion overrides": ", ".join(f"{k}={v}" for k, v in p.get("fusion", {}).items()) or "defaults",
                            "weights present": (PROJECT_ROOT / p["weapon_model"]).exists(),
                            "evaluated in": p.get("evaluated_in", "")} for n, p in list_profiles().items()]),
             hide_index=True, width="stretch")

with st.expander("Available weapon detectors"):
    st.dataframe(pd.DataFrame([{"name": k, "weights": str(v.relative_to(PROJECT_ROOT))} for k, v in detector_registry().items()]),
                 hide_index=True, width="stretch")
with st.expander("Learned weapon-action head"):
    st.json(head)

st.subheader("Headline results (stored research runs)")
comparisons = sorted(RUNS.glob("*_profile-comparison/results.json"))
if comparisons:   # latest profile comparison: every profile on the same recordings, held-out data
    cmp_ = json.loads(comparisons[-1].read_text())
    rows = []
    for name, r in cmp_["results"].items():
        t, ci = r["test"], r["test"]["usrt_ci"]
        vr = ci.get("verified_event_recall_ci95", [None, None])
        rows.append({"profile": name, "Cam7 verified recall": t["usrt"]["verified_event_recall"],
                     "95% CI": f"{vr[0]}–{vr[1]}", "Cam7 FA/min": t["usrt"]["false_alarms_per_min_outside_gt"],
                     "vtest alarms/min (weapon-free)": t["vtest_alarms_per_min"],
                     "Mendeley test recall": t["mendeley"]["event_recall"], "Mendeley FA/min": t["mendeley"]["fa_per_min"],
                     "UCF test recall": t["ucf"]["event_recall"], "UCF FA/min": t["ucf"]["fa_per_min"]})
    st.caption(f"Held-out data, alarm = level ≥ SUSPICIOUS_ACTIVITY, spatially verified on mock-attack Cam7 "
               f"(`{comparisons[-1].parent.name}`).")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
else:
    st.info("No profile comparison yet (research/tools/integration/evaluate_profiles.py).")
reports = [r for r in list_runs() if r["report"] and not r["aborted"]]
if reports:
    st.markdown("**Latest reports:** " + " · ".join(f"`{r['name']}`" for r in reports[:4]) + " — open them under *Research results*.")

st.subheader("Background jobs on this machine")
jobs = active_jobs()
st.code("\n".join(jobs) if jobs else "none", language="text")

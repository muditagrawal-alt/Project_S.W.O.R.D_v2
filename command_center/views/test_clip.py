"""Test a clip: run the connected system (Phases 1-5) on any source with any configuration and watch it live.

Every run is stored as a research run (research/runs/<time>_command-centre-<source>) through pipeline.run.run_video, so
results are reproducible and include the baseline-ladder evaluation when ground truth exists (mock-attack sequences).
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import LEVELS, PROJECT_ROOT, RUNS, UPLOADS, detector_registry, sample_sources  # noqa: E402
from fusion.rules import FusionConfig  # noqa: E402
from pipeline.profiles import default_profile, list_profiles  # noqa: E402

st.title("🎬 Test a clip")
st.caption("Runs Phases 1→5 in this process on the Apple GPU (MPS). The run is saved under research/runs/.")

# ───────────────────────────── source ─────────────────────────────
left, right = st.columns([3, 2])
with left:
    kind = st.radio("Source", ["Built-in test clip", "Upload a video", "Camera"], horizontal=True)
    source, gt, live = None, None, False
    if kind == "Built-in test clip":
        samples = sample_sources()
        if samples:
            choice = st.selectbox("Clip", list(samples))
            source = samples[choice]
            gt = source if source.suffix == ".json" else None
        else:   # fresh clone: research/data/ is not versioned; uploads and cameras still work
            st.warning("No local test clips (research/data/ is not in git). Upload a video or use a camera instead.")
    elif kind == "Upload a video":
        up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
        if up is not None:
            UPLOADS.mkdir(parents=True, exist_ok=True)
            source = UPLOADS / f"{datetime.now():%Y%m%d-%H%M%S}_{Path(up.name).name}"
            source.write_bytes(up.getbuffer())
    else:
        cam = st.number_input("Camera index", 0, 8, 0)
        source, live = str(int(cam)), True
with right:
    max_frames = st.number_input("Max frames (0 = all)", 0, 100000, 300 if kind != "Built-in test clip" else 0, step=50)
    stride = st.number_input("Frame stride (process every k-th frame)", 1, 10, 1)
    update_every = st.slider("Refresh the view every N frames", 1, 30, 5)

# ───────────────────────────── configuration ─────────────────────────────
ZOOMS = ["off", "person", "hands", "person+hands"]
with st.expander("Configuration", expanded=True):
    profiles = list_profiles()
    pnames = list(profiles)
    prof = st.selectbox("Profile (an evaluated configuration; change any option below to experiment)", pnames,
                        index=pnames.index(default_profile()) if default_profile() in pnames else 0)
    P = profiles[prof]
    st.caption(f"{P.get('description', '')}  ·  evaluated in `{P.get('evaluated_in') or '—'}`")
    reg = detector_registry()
    prof_det = PROJECT_ROOT / P["weapon_model"]
    if prof_det.exists() and prof_det.resolve() not in {v.resolve() for v in reg.values()}:
        reg = {f"{prof} profile ({prof_det.name})": prof_det} | reg
    names = list(reg)
    det_default = next((i for i, k in enumerate(names) if reg[k].resolve() == prof_det.resolve()), 0)
    fz = FusionConfig() if not P.get("fusion") else FusionConfig(**{k: v for k, v in P["fusion"].items()
                                                                     if k in FusionConfig.__dataclass_fields__})
    zoom_default = "+".join(P.get("weapon_zoom", [])) or "off"
    k = f"_{prof}"   # widget keys per profile: switching profile resets the options to that profile's values
    c1, c2, c3 = st.columns(3)
    with c1:
        det_name = st.selectbox("Weapon detector (Phase 1)", names, index=det_default, key="det" + k)
        imgsz = st.select_slider("Detector input size", [640, 960, 1280], value=int(P.get("weapon_imgsz", 640)), key="sz" + k)
        zoom = st.selectbox("Person-zoom crops", ZOOMS, index=ZOOMS.index(zoom_default) if zoom_default in ZOOMS else 0,
                            key="zoom" + k)
        zoom_min = st.slider("Minimum score for zoom-crop detections", 0.0, 0.6, float(P.get("zoom_min_score", 0.0)), 0.05,
                             key="zmin" + k, disabled=zoom == "off")
        min_window = st.select_slider("Skip zoom windows smaller than (px)", [0, 64, 96, 128, 160],
                                      value=int(P.get("zoom_min_window", 0)), key="zwin" + k, disabled=zoom == "off",
                                      help="96 px = the smallest crop the zoom detector saw in training")
        max_area_pct = st.select_slider("Drop boxes larger than (% of frame)", [5.0, 8.5, 12.5, 25.0, 100.0],
                                        value=round(100 * float(P.get("weapon_max_area", 1.0)), 1), key="area" + k,
                                        help="largest CCTV training box: 12.3% of the frame; 100 = off")
    with c2:
        use_pose = st.checkbox("Pose (Phase 3)", True)
        use_action = st.checkbox("Action recognition (Phase 4)", True, disabled=not use_pose)
        use_head = st.checkbox("Learned weapon-action head", True, disabled=not (use_pose and use_action))
        async_action = st.checkbox("Phase 4 on a worker thread (live)", live)
    with c3:
        hold_s = st.slider("Hold time before a level drops (s)", 0.5, 5.0, float(fz.hold_s), 0.5, key="hold" + k)
        persist_s = st.slider("Weapon persistence for WEAPON_RELATED_EVENT (s)", 0.0, 3.0, float(fz.persist_s), 0.5,
                              key="persist" + k)
        w_susp = st.slider("Weapon evidence threshold (suspicious)", 0.2, 0.9, float(fz.w_susp), 0.05, key="wsusp" + k)
        w_event = st.slider("Weapon evidence threshold (event)", 0.2, 0.9, float(fz.w_event), 0.05, key="wevent" + k)
        l_susp = st.slider("Learned head alone raises SUSPICIOUS at", 0.5, 1.01, float(fz.l_susp), 0.01, key="lsusp" + k,
                           help="1.01 = never on its own")
        a_susp = st.slider("PoseC3D violent alone raises SUSPICIOUS at", 0.3, 1.0, float(fz.a_susp), 0.05, key="asusp" + k)
    save_video = st.checkbox("Save annotated video", True)

run = st.button("▶ Run", type="primary", disabled=source is None)


@st.cache_resource(show_spinner="Loading models (first run only)…", max_entries=2)
def get_system(fps: float, det_path: str, imgsz: int, zoom: str, zoom_min: float, min_window: int, max_area: float,
               use_pose: bool, use_action: bool, async_action: bool):
    from pipeline.system import SwordSystem

    return SwordSystem(fps=fps, weapon_model=det_path, weapon_imgsz=imgsz, use_pose=use_pose, use_action=use_action,
                       async_action=async_action, weapon_zoom=tuple(m for m in zoom.split("+") if m != "off"),
                       zoom_min_score=zoom_min, zoom_min_window=min_window, weapon_max_area=max_area)


if run:
    import cv2

    from fusion.rules import TemporalThreatEngine
    from pipeline.run import draw, open_source, run_video

    cap = open_source(str(source))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / stride
    cap.release()
    zoom_on = zoom != "off"
    system = get_system(round(fps, 3), str(reg[det_name]), imgsz, zoom, zoom_min if zoom_on else 0.0,
                        min_window if zoom_on else 0, max_area_pct / 100, use_pose, use_action and use_pose,
                        async_action and use_action)
    overrides = {k: v for k, v in P.get("fusion", {}).items() if k in FusionConfig.__dataclass_fields__}
    cfg = FusionConfig.for_fps(fps, **overrides | dict(hold_s=hold_s, persist_s=persist_s, w_susp=w_susp, w_event=w_event,
                                                     l_susp=l_susp, a_susp=a_susp, use_learned_action=use_head))
    system.p5 = TemporalThreatEngine(cfg)
    tag = (Path(str(source)).stem if not live else f"camera{source}")[:40].replace(" ", "_")
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}_command-centre-{tag}"

    banner, view = st.empty(), st.empty()
    m1, m2, m3, m4 = (col.empty() for col in st.columns(4))   # placeholders: each refresh replaces the metric
    chart, events_box, timing_box = st.empty(), st.empty(), st.empty()
    hist = {"t": [], "level": []}
    events = []
    t_start = time.time()

    def on_frame(rec, frame):
        hist["t"].append(rec["timestamp"])
        hist["level"].append(LEVELS.index(rec["threat"]["scene_level"]))
        events.extend(rec["fusion_events"])
        n = len(hist["t"])
        if n % update_every and not rec["fusion_events"]:
            return
        lv = rec["threat"]["scene_level"]
        colour = {"NORMAL": "green", "ELEVATED": "orange", "SUSPICIOUS_ACTIVITY": "red", "WEAPON_RELATED_EVENT": "violet"}[lv]
        banner.markdown(f"### :{colour}[{lv.replace('_', ' ')}]  ·  t = {rec['timestamp']:.1f} s  ·  frame {rec['frame_idx']}")
        view.image(draw(frame, rec)[:, :, ::-1], width="stretch")
        m1.metric("People tracked", len(rec["persons"]))
        m2.metric("Weapon detections", len(rec["weapon_detections"]))
        m3.metric("Frames processed", n)
        m4.metric("Throughput", f"{n / max(time.time() - t_start, 1e-6):.1f} fps")
        chart.line_chart(pd.DataFrame({"scene level (0 NORMAL … 3 WEAPON_RELATED_EVENT)": hist["level"]}, index=hist["t"]),
                         height=180)
        if events:
            events_box.dataframe(pd.DataFrame([{k: e.get(k) for k in ("t", "scope", "track_id", "from", "to")} for e in events[-12:]]),
                                 hide_index=True, width="stretch")

    with st.spinner("Running…"):
        res = run_video(str(source), out, system=system, gt=str(gt) if gt else None, save_video=save_video and not live,
                        max_frames=int(max_frames) or None, on_frame=on_frame, live=live, keep_frames=not live,
                        frame_stride=int(stride) if not str(source).endswith(".json") else 1)
    s = res["summary"]
    st.success(f"Done: {s['frames']} frames in {s['wall_seconds']} s — saved to research/runs/{out.name}")
    run_cfg = {"source": str(source), "profile": prof, "detector": det_name, "imgsz": imgsz, "zoom": zoom,
               "zoom_min_score": zoom_min, "zoom_min_window": min_window, "weapon_max_area": max_area_pct / 100, "pose": use_pose, "action": use_action, "learned_head": use_head,
               "fusion": cfg.__dict__ | {"features": None}}
    (out / "command_centre_config.json").write_text(json.dumps(run_cfg, indent=2, default=str))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Time spent in each scene level**")
        st.bar_chart(pd.Series(s["scene_level_share"]).reindex(LEVELS).fillna(0) * 100, height=220, sort=False)
    with c2:
        st.markdown("**Per-stage latency (ms per frame)**")
        t = {k: v["mean"] for k, v in s["timing_ms"].items() if isinstance(v, dict)}
        st.bar_chart(pd.Series(t), height=220)
    if res["evaluation"]:
        st.markdown("**Baseline ladder on this clip** (verified columns need ground truth)")
        ev = {k: v for k, v in res["evaluation"].items() if not k.startswith("_")}
        cols = ["alarm_segments", "alarms_per_min", "time_in_alarm_pct", "event_recall", "verified_event_recall",
                "false_alarms_per_min_outside_gt", "misattributed_alarm_segments"]
        st.dataframe(pd.DataFrame(ev).T[[c for c in cols if c in pd.DataFrame(ev).T.columns]], width="stretch")
    if save_video and (out / "annotated.mp4").exists():
        from service.incidents import ffmpeg_exe
        import subprocess

        h264 = out / "annotated_h264.mp4"
        exe = ffmpeg_exe()
        if exe:
            subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(out / "annotated.mp4"), "-c:v", "libx264", "-pix_fmt",
                            "yuv420p", "-movflags", "+faststart", str(h264)])
        st.video(str(h264 if h264.exists() else out / "annotated.mp4"))

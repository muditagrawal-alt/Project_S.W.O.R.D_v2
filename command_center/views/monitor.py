"""Live monitor: the Phase 6 service (runs, live preview, incidents with evidence, audit log) and training/jobs."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import (LEVELS, ServiceClient, active_jobs, sample_sources,  # noqa: E402
                 start_service_background, training_curves)
from pipeline.profiles import default_profile, list_profiles  # noqa: E402

st.title("📡 Live monitor")
url = st.text_input("Service URL", st.session_state.get("service_url", "http://127.0.0.1:8000"))
st.session_state["service_url"] = url
client = ServiceClient(url)
actor = st.text_input("Operator name (recorded in the audit log)", st.session_state.get("actor", "operator"))
st.session_state["actor"] = actor

tab_service, tab_training = st.tabs(["Incident service (Phase 6)", "Training & jobs"])

with tab_service:
    health = client.health()
    if not health:
        st.warning("The service is not running at this address.")
        if st.button("Start the service on 127.0.0.1:8000"):
            start_service_background(8000)
            st.info("Starting… (models load on the first run). Refresh in a few seconds.")
    else:
        st.caption(f"database: {health['database']} · FFmpeg: {'yes' if health['ffmpeg'] else 'no'} · data: {health['data_dir']}")

        with st.expander("Start a run"):
            samples = {k: v for k, v in sample_sources().items()}
            src_kind = st.radio("Source", ["Built-in clip", "Path / camera index / stream URL"], horizontal=True)
            source = str(samples[st.selectbox("Clip", list(samples))]) if src_kind == "Built-in clip" else st.text_input("Source", "0")
            profiles = list_profiles()
            c1, c2, c3, c4 = st.columns(4)
            prof = c1.selectbox("Profile", list(profiles), index=list(profiles).index(default_profile())
                                if default_profile() in profiles else 0,
                                help="detector, input size, zoom and fusion settings of an evaluated configuration")
            open_level = c2.selectbox("Open an incident at", LEVELS[1:], index=1)
            c3.caption(f"{profiles[prof].get('weapon_model')} @ {profiles[prof].get('weapon_imgsz')} px · zoom: "
                       f"{'+'.join(profiles[prof].get('weapon_zoom', [])) or 'off'}")
            live = c4.checkbox("Live mode (wall-clock time, async Phase 4)", False)
            if st.button("Start run", type="primary"):
                body = {"source": source, "actor": actor, "open_level": open_level, "live": live, "profile": prof}
                try:
                    r = client.start_run(body)
                    st.success(f"Run {r['id']} queued")
                except Exception as e:   # noqa: BLE001
                    st.error(str(e))

        @st.fragment(run_every="2s")
        def live_panel():
            runs = client.runs()
            active = [r for r in runs if r["status"] in ("running", "queued")]
            c1, c2 = st.columns([3, 2])
            with c1:
                if active:
                    r = active[0]
                    img = client.preview(r["id"])
                    st.markdown(f"**Live: run `{r['id']}`** — {Path(r['source']).name} ({r['status']})")
                    if img:
                        st.image(img, width="stretch")
                    if st.button("Stop this run", key=f"stop-{r['id']}"):
                        client.stop_run(r["id"], actor)
                else:
                    st.info("No active run.")
            with c2:
                inc = client.incidents(limit=10)
                st.markdown("**Latest incidents**")
                if inc:
                    st.dataframe(pd.DataFrame([{"#": i["id"], "peak": i["level_peak"], "start s": round(i["start_t"], 1),
                                                "status": i["status"], "ack": i["ack_by"] or ""} for i in inc]),
                                 hide_index=True, width="stretch")
                st.markdown("**Runs**")
                st.dataframe(pd.DataFrame([{"run": r["id"], "status": r["status"], "frames": r["frames"],
                                            "incidents": (r.get("summary") or {}).get("incidents")} for r in runs[:8]]),
                             hide_index=True, width="stretch")

        live_panel()

        st.subheader("Incident review")
        incidents = client.incidents(limit=200)
        if incidents:
            pick = st.selectbox("Incident", [f"#{i['id']} · {i['level_peak']} · run {i['run_id']} · {i['start_t']:.1f}s"
                                             for i in incidents])
            iid = int(pick.split()[0][1:])
            d = client.incident(iid)
            c1, c2 = st.columns([3, 2])
            with c1:
                if d.get("clip_path"):
                    st.video(client.media(f"/api/incidents/{iid}/clip", actor=actor))
                    st.caption(f"clip SHA-256 {d.get('clip_sha256')}")
                else:
                    st.info("The evidence clip is written when the incident closes.")
            with c2:
                if d.get("snapshot_path"):
                    st.image(client.media(f"/api/incidents/{iid}/snapshot", actor=actor), caption="annotated snapshot")
                st.json({k: d[k] for k in ("level_start", "level_peak", "start_t", "end_t", "track_ids", "evidence", "status",
                                           "ack_by", "ack_note") if k in d})
                note = st.text_input("Review note")
                if st.button("Acknowledge"):
                    client.ack(iid, actor, note)
                    st.success("Acknowledged (recorded in the audit log)")
            if d.get("events"):
                st.dataframe(pd.DataFrame([{k: e[k] for k in ("t", "scope", "track_id", "from_level", "to_level")} for e in d["events"]]),
                             hide_index=True, width="stretch")
        with st.expander("Audit log"):
            st.dataframe(pd.DataFrame(client.audit(300)), hide_index=True, width="stretch")


with tab_training:
    @st.fragment(run_every="10s")
    def training_panel():
        curves = training_curves()
        if curves:
            name = st.selectbox("Detector training run", list(curves)[::-1])
            df = pd.read_csv(curves[name])
            df.columns = [c.strip() for c in df.columns]
            cols = [c for c in df.columns if c.startswith("metrics/")]
            st.line_chart(df.set_index("epoch")[cols], height=260)
            st.dataframe(df.tail(5), hide_index=True, width="stretch")
        st.markdown("**Background jobs**")
        jobs = active_jobs()
        st.code("\n".join(jobs) if jobs else "none", language="text")

    training_panel()

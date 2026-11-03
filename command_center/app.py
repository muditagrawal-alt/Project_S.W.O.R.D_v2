"""
S.W.O.R.D. command centre — one Streamlit app for testing and monitoring the connected system.

    streamlit run command_center/app.py

Pages: overview (system + headline results), test a clip (Phases 1-5 in-process with any configuration), live monitor
(the Phase 6 service: runs, incidents with evidence, audit log, training/jobs), research results (every stored run),
detector lab (the team's original app.py, unchanged) and labelling (the interval-label review tool).
"""

from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="S.W.O.R.D. command centre", page_icon="🛡️", layout="wide")

pages = {
    "Command centre": [
        st.Page(HERE / "views" / "overview.py", title="Overview", icon="🛡️", default=True),
        st.Page(HERE / "views" / "test_clip.py", title="Test a clip", icon="🎬"),
        st.Page(HERE / "views" / "monitor.py", title="Live monitor", icon="📡"),
        st.Page(HERE / "views" / "results.py", title="Research results", icon="📊"),
    ],
    "Tools": [
        st.Page(HERE.parent / "app.py", title="Detector lab (team app)", icon="🔫", url_path="detector_lab"),
        st.Page(HERE / "views" / "labeling.py", title="Interval labelling", icon="🏷️"),
    ],
}
st.navigation(pages).run()

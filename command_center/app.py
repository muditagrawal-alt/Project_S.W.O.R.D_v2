"""
S.W.O.R.D. command centre — one Streamlit app for testing and monitoring the connected system.

    streamlit run command_center/app.py

Pages: overview (system + headline results), test a clip (Phases 1-5 in-process with any configuration), live monitor
(the Phase 6 service: runs, incidents with evidence, audit log, training/jobs), research results (every stored run),
detector lab (the team's original app.py, unchanged) and labelling (the interval-label review tool).
"""

import os
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
st.set_page_config(page_title="S.W.O.R.D. command centre", page_icon="🛡️", layout="wide")

# Streamlit reloads an edited module only if it lives next to this script or on PYTHONPATH. With the project root on
# PYTHONPATH, edits to pipeline/, fusion/, service/... are picked up on the next rerun, instead of the running app
# keeping the old module and failing with "ImportError: cannot import name ...". Streamlit's folder blacklist keeps
# venv/ and site-packages out of the watch list.
_pythonpath = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
if str(PROJECT_ROOT) not in _pythonpath:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT)] + _pythonpath)

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lib import missing_packages  # noqa: E402

_missing = missing_packages()
if _missing:
    st.error(f"This Python cannot run S.W.O.R.D.: it is missing **{', '.join(_missing)}**.\n\n"
             f"It is `{sys.executable}`, not the project's virtual environment. Stop this app and start it with:\n\n"
             "```\nsource venv/bin/activate\nstreamlit run command_center/app.py\n```\n"
             "(or install the dependencies into this Python: `pip install -r requirements.txt`).")
    st.stop()

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

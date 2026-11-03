"""Interval labelling: the review tool lives in the Phase 6 service (/labeler); this page links to it and shows status."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import DATA, ServiceClient, start_service_background  # noqa: E402

st.title("🏷️ Interval labelling")
st.caption("Optional: verify the derived interval labels used to train and evaluate Phase 5. Rules: fusion/LABELING_GUIDELINES.md.")
url = st.session_state.get("service_url", "http://127.0.0.1:8000")
if ServiceClient(url).health():
    st.link_button("Open the labelling tool", f"{url}/labeler", type="primary")
else:
    st.warning("The service is not running.")
    if st.button("Start the service"):
        start_service_background(8000)
        st.info("Starting… refresh in a few seconds.")

rows = []
for f in sorted((DATA / "labels").glob("*.csv")):
    df = pd.read_csv(f)
    rows.append({"file": f.name, "videos": df["video"].nunique(), "intervals": len(df),
                 "human-verified": int(df["source"].astype(str).str.startswith("human").sum())})
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

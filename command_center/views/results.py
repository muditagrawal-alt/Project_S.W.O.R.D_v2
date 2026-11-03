"""Research results: browse every stored run (reports, tables, figures, ladders) and compare configurations."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import list_runs  # noqa: E402

st.title("📊 Research results")
st.caption("Everything under research/runs/ (local, not versioned). Aborted, superseded and smoke runs are hidden by default.")

runs = list_runs()
show_all = st.toggle("Show aborted / superseded / smoke runs", False)
runs = [r for r in runs if show_all or not r["aborted"]]
kinds = sorted({r["kind"].split("-")[0] for r in runs})
c1, c2 = st.columns([1, 3])
with c1:
    kind = st.selectbox("Type", ["all"] + kinds)
    filtered = [r for r in runs if kind == "all" or r["kind"].startswith(kind)]
    name = st.radio("Run", [r["name"] for r in filtered], label_visibility="collapsed") if filtered else None
with c2:
    if not name:
        st.info("No runs.")
        st.stop()
    run = next(r for r in filtered if r["name"] == name)
    d = run["path"]
    st.subheader(name)
    tabs = st.tabs(["Report", "Tables", "Ladder", "Figures", "Files"])
    with tabs[0]:
        if (d / "REPORT.md").exists():
            st.markdown((d / "REPORT.md").read_text())
        else:
            st.info("No REPORT.md in this run.")
    with tabs[1]:
        for t in sorted(d.glob("*table*.md")) + sorted(d.glob("*/results_table.md")):
            st.markdown(f"**{t.relative_to(d)}**")
            st.markdown(t.read_text())
        if not list(d.glob("*table*.md")) and not list(d.glob("*/results_table.md")):
            st.info("No result tables.")
    with tabs[2]:
        res = d / "results.json"
        data = json.loads(res.read_text()) if res.exists() else {}
        splits = data.get("splits") if isinstance(data, dict) else None
        if splits:
            split = st.selectbox("Split", list(splits))
            rows = []
            for rung, m in splits[split].items():
                rows.append({"rung": rung, "verified recall": m.get("verified_event_recall"), "recall (time-only)": m.get("event_recall"),
                             "FA/min outside GT": m.get("false_alarms_per_min_outside_gt"), "alarms/min": m.get("alarms_per_min"),
                             "misattributed": m.get("misattributed_alarm_segments")})
            df = pd.DataFrame(rows).set_index("rung")
            st.dataframe(df, width="stretch")
            plot = df[["verified recall", "FA/min outside GT"]].dropna()
            if not plot.empty:
                st.scatter_chart(plot.reset_index(), x="FA/min outside GT", y="verified recall", color="rung", height=320)
        else:
            st.info("This run has no baseline-ladder results.")
    with tabs[3]:
        figs = sorted(d.glob("*.png"))
        for f in figs:
            st.image(str(f), caption=f.name, width="stretch")
        if not figs:
            st.info("No figures.")
    with tabs[4]:
        files = sorted(p for p in d.rglob("*") if p.is_file() and p.stat().st_size < 50_000_000)[:400]
        st.dataframe(pd.DataFrame([{"file": str(p.relative_to(d)), "KB": round(p.stat().st_size / 1024, 1)} for p in files]),
                     hide_index=True, width="stretch", height=360)
        js = [p for p in files if p.suffix == ".json" and p.stat().st_size < 2_000_000]
        if js:
            pick = st.selectbox("Show a JSON file", [str(p.relative_to(d)) for p in js])
            st.json(json.loads((d / pick).read_text()), expanded=False)

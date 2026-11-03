"""
Rebuild research/RUNS_INDEX.csv: one row per folder in research/runs/ with its date, kind, status (ok / ABORTED /
SUPERSEDED / SMOKE, from the folder name), the metadata and report files it contains, the git commit recorded in its
manifest (when any) and a one-line description.

Descriptions come from, in order: the run's manifest.json ("description"), research/run_descriptions.json (one
entry per run, for runs whose tools did not write a description) and the runs table in research/README.md (rows
that name the folder). Runs left without a description are listed at the end so they can be labelled.

    python research/tools/index_runs.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
RUNS = RESEARCH / "runs"
META_FILES = ("manifest.json", "results.json", "REPORT.md", "summary.json", "results_table.md", "table.md")


def readme_rows() -> list:
    readme = RESEARCH / "README.md"
    if not readme.exists():
        return []
    rows = []
    for line in readme.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 2 and "`" in cells[0]:
            rows.append((cells[0], " — ".join(c for c in cells[1:] if c and set(c) != {"-"})))
    return rows


def from_readme(run: str, rows: list) -> str:
    for names, text in rows:
        for token in re.findall(r"`([^`]+)`", names):
            token = token.replace("*", "")
            if token and (run == token or (token.endswith("...") and run.startswith(token[:-3]))
                          or ("{" in token and re.fullmatch(_brace_regex(token), run))):
                return text
    return ""


def _brace_regex(token: str) -> str:
    out, i = "", 0
    for m in re.finditer(r"\{([^}]+)\}", token):
        out += re.escape(token[i:m.start()]) + "(" + "|".join(map(re.escape, m.group(1).split(","))) + ")"
        i = m.end()
    return out + re.escape(token[i:])


def main():
    overlay_path = RESEARCH / "run_descriptions.json"
    overlay = json.loads(overlay_path.read_text()) if overlay_path.exists() else {}
    rows, missing = [], []
    readme = readme_rows()
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        name = d.name
        m = re.match(r"(\d{8})(?:-(\d{6}))?[_-]?(.*)", name)
        date, time_, kind = (m.group(1), m.group(2) or "", m.group(3)) if m else ("", "", name)
        status = next((s for s in ("ABORTED", "FAILED", "SUPERSEDED", "SMOKE") if s in name.upper()), "ok")
        manifest = {}
        if (d / "manifest.json").exists():
            try:
                manifest = json.loads((d / "manifest.json").read_text())
            except json.JSONDecodeError:
                manifest = {}
        desc = (manifest.get("description") if isinstance(manifest, dict) else None) or overlay.get(name) or from_readme(name, readme)
        if not desc:
            missing.append(name)
        rows.append({"run_id": name, "date": date, "time": time_, "kind": kind, "status": status,
                     "metadata_files": " ".join(f for f in META_FILES if (d / f).exists()),
                     "git_commit": (manifest.get("git_head") or manifest.get("git_commit") or "") if isinstance(manifest, dict) else "",
                     "description": desc or ""})
    out = RESEARCH / "RUNS_INDEX.csv"
    with open(out, "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} runs -> {out}")
    if missing:
        print(f"{len(missing)} runs without a description:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()

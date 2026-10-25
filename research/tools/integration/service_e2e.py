"""
Phase 6 end-to-end test against the REAL system: uvicorn + FastAPI + WebSocket + SQLite + FFmpeg, Phases 1-5 live.

Starts the service in a subprocess (data and database inside --out), listens on /ws/events, submits runs over HTTP,
waits for them, then verifies: incidents exist and are closed, every evidence clip downloads and matches the
SHA-256 stored in the database, clips are H.264, acknowledgement and evidence access are in the audit log,
WebSocket message counts per type. Writes report.json + REPORT.md; the database and evidence stay in --out.

    python research/tools/integration/service_e2e.py --out research/runs/<id>_phase6-service-e2e \\
        --run '{"source": "research/data/videos/vtest.avi", "open_level": "ELEVATED"}' \\
        --run '{"source": "research/data/videos/vtest.avi", "live": true, "max_frames": 300}'
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def listen(url: str, sink: list, stop: asyncio.Event):
    import websockets

    async with websockets.connect(url, max_size=None) as ws:
        while not stop.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            sink.append((time.time(), json.loads(msg)))


async def main_async(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = dict(os.environ, SWORD_DATA_DIR=str(out / "service_data"), SWORD_DATABASE_URL=f"sqlite:///{out / 'sword.db'}",
               SWORD_MEDIA_ROOTS=str(PROJECT_ROOT / "research" / "data"))
    log = open(out / "server_log.txt", "w")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "service.api:create_app", "--host", "127.0.0.1",
                               "--port", str(port)], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    report = {"created": datetime.now().isoformat(timespec="seconds"), "runs": [], "checks": {}}
    msgs: list = []
    stop = asyncio.Event()
    try:
        async with httpx.AsyncClient(base_url=base, timeout=30) as c:
            for _ in range(120):
                try:
                    if (await c.get("/api/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
            report["health"] = (await c.get("/api/health")).json()
            report["checks"]["dashboard_served"] = "S.W.O.R.D." in (await c.get("/")).text
            listener = asyncio.create_task(listen(f"ws://127.0.0.1:{port}/ws/events", msgs, stop))
            await asyncio.sleep(0.5)
            for spec in args.run:
                body = json.loads(spec)
                body["source"] = str((PROJECT_ROOT / body["source"]).resolve())
                body.setdefault("actor", "e2e-test")
                t0 = time.time()
                r = await c.post("/api/runs", json=body)
                assert r.status_code == 201, r.text
                run_id = r.json()["id"]
                while True:
                    run = (await c.get(f"/api/runs/{run_id}")).json()
                    if run["status"] in ("finished", "failed", "stopped"):
                        break
                    await asyncio.sleep(1.0)
                incs = (await c.get("/api/incidents", params={"run_id": run_id})).json()
                checked = []
                for inc in incs:
                    item = {"id": inc["id"], "level_start": inc["level_start"], "level_peak": inc["level_peak"],
                            "start_t": inc["start_t"], "end_t": inc["end_t"], "status": inc["status"],
                            "track_ids": inc["track_ids"], "evidence": inc["evidence"]}
                    if inc["clip_path"]:
                        clip = await c.get(f"/api/incidents/{inc['id']}/clip", params={"actor": "e2e-reviewer"})
                        item["clip_bytes"] = len(clip.content)
                        item["clip_sha256_matches_db"] = hashlib.sha256(clip.content).hexdigest() == inc["clip_sha256"]
                        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                "stream=codec_name,width,height,nb_frames", "-of", "json", inc["clip_path"]],
                                               capture_output=True, text=True)
                        item["clip_stream"] = (json.loads(probe.stdout or "{}").get("streams") or [{}])[0]
                    detail = (await c.get(f"/api/incidents/{inc['id']}")).json()
                    item["events"] = len(detail["events"])
                    checked.append(item)
                if incs:
                    ack = await c.post(f"/api/incidents/{incs[-1]['id']}/ack", json={"actor": "e2e-reviewer", "note": "e2e check"})
                    report["checks"].setdefault("ack_ok", True)
                    report["checks"]["ack_ok"] &= ack.status_code == 200 and ack.json()["ack_by"] == "e2e-reviewer"
                report["runs"].append({"request": body, "run_id": run_id, "status": run["status"], "error": run.get("error"),
                                       "wall_s": round(time.time() - t0, 1), "summary": run.get("summary"),
                                       "incidents": checked})
            await asyncio.sleep(1.0)
            stop.set()
            await listener
            audit = (await c.get("/api/audit", params={"limit": 5000})).json()
        types = Counter(m["type"] for _, m in msgs)
        actions = Counter(a["action"] for a in audit)
        report["websocket_messages"] = dict(types)
        report["audit_actions"] = dict(actions)
        all_incs = [i for r in report["runs"] for i in r["incidents"]]
        report["checks"].update(
            all_runs_finished=all(r["status"] == "finished" for r in report["runs"]),
            incidents_total=len(all_incs),
            all_incidents_closed=all(i["status"] == "closed" for i in all_incs),
            all_clips_hash_verified=all(i.get("clip_sha256_matches_db") for i in all_incs) if all_incs else None,
            all_clips_h264=all(i.get("clip_stream", {}).get("codec_name") == "h264" for i in all_incs) if all_incs else None,
            ws_frames_received=types.get("frame", 0) > 0,
            ws_incident_messages_match=types.get("incident_opened", 0) == len(all_incs),
            audit_has_evidence_access=actions.get("evidence_clip_accessed", 0) == sum(1 for i in all_incs if "clip_bytes" in i),
            audit_has_run_lifecycle=actions.get("run_created", 0) == len(report["runs"]) == actions.get("run_finished", 0))
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    lines = [f"# Phase 6 service end-to-end — {report['created']}", "", "## Checks", ""]
    lines += [f"- {k}: **{v}**" for k, v in report["checks"].items()]
    lines += ["", "## Runs", "", "| source | live | status | frames | throughput fps | incidents (peak levels) | wall s |", "|---|---|---|---|---|---|---|"]
    for r in report["runs"]:
        s = r["summary"] or {}
        peaks = Counter(i["level_peak"] for i in r["incidents"])
        lines.append(f"| {Path(r['request']['source']).name} | {s.get('live')} | {r['status']} | {s.get('frames')} | "
                     f"{(s.get('timing_ms') or {}).get('throughput_fps')} | {len(r['incidents'])} {dict(peaks)} | {r['wall_s']} |")
    lines += ["", f"WebSocket messages: {report['websocket_messages']}", "", f"Audit actions: {report['audit_actions']}"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", required=True, help="JSON body for POST /api/runs (source relative to the repo)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()

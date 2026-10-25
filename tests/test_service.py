"""Phase 6 service: runs, incidents, evidence clips + hashes, events, audit log, WebSocket — with a fake Phases 1-5."""

import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fusion.rules import LEVELS, TemporalThreatEngine  # noqa: E402
from service.api import create_app  # noqa: E402

FPS = 10
# frame ranges -> scene level: rise to SUSPICIOUS at 15, escalate to WRE at 30, back to NORMAL at 35
SCHEDULE = [(0, 15, "NORMAL"), (15, 30, "SUSPICIOUS_ACTIVITY"), (30, 35, "WEAPON_RELATED_EVENT"), (35, 60, "NORMAL")]


def level_at(i):
    return next(lv for a, b, lv in SCHEDULE if a <= i < b)


class FakeSystem:
    sources = {"phase1": "fake"}

    def __init__(self):
        self.p5 = TemporalThreatEngine()
        self.prev = "NORMAL"

    def reset(self):
        self.prev = "NORMAL"

    def timing_summary(self):
        return {"total_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "throughput_fps": 1000.0}

    def process(self, frame, idx, t):
        lv = level_at(idx)
        events = []
        if lv != self.prev:
            events = [{"type": "level_change", "scope": "track", "track_id": 1, "t": round(t, 3), "from": self.prev, "to": lv,
                       "evidence": {"weapon_hand": 0.6}},
                      {"type": "level_change", "scope": "scene", "track_id": None, "t": round(t, 3), "from": self.prev,
                       "to": lv, "tracks_at_level": [1]}]
        self.prev = lv
        ev = {"weapon_hand": 0.6 if lv != "NORMAL" else 0.0, "act_weapon": 0.0}
        return {"frame_idx": idx, "timestamp": round(t, 4), "persons": [], "weapons": [], "weapon_detections": [],
                "actions": {}, "events": [], "fusion_events": events,
                "threat": {"scene_level": lv, "weapon_unattended_ema": 0.0, "tracks": {1: {"level": lv, "evidence": ev}}}}


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "media" / "clip.avi"
    path.parent.mkdir()
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for i in range(60):
        f = np.full((48, 64, 3), (i * 4) % 255, np.uint8)
        cv2.putText(f, str(i), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        wr.write(f)
    wr.release()
    return path


@pytest.fixture()
def client(tmp_path, video):
    app = create_app(database_url=f"sqlite:///{tmp_path / 'test.db'}", data_dir=str(tmp_path / "data"),
                     media_roots=[str(video.parent)], make_system=lambda fps, req, live: FakeSystem())
    with TestClient(app) as c:
        yield c


def wait_run(c, run_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/runs/{run_id}").json()
        if r["status"] in ("finished", "failed", "stopped"):
            return r
        time.sleep(0.1)
    raise TimeoutError(run_id)


def test_run_creates_incident_with_evidence_and_audit(client, video):
    r = client.post("/api/runs", json={"source": str(video), "actor": "tester", "pre_s": 1.0, "post_s": 1.0})
    assert r.status_code == 201, r.text
    run = wait_run(client, r.json()["id"])
    assert run["status"] == "finished", run.get("error")
    assert run["frames"] == 60 and run["summary"]["incidents"] == 1

    (inc,) = client.get("/api/incidents", params={"run_id": run["id"]}).json()
    assert inc["level_start"] == "SUSPICIOUS_ACTIVITY" and inc["level_peak"] == "WEAPON_RELATED_EVENT"
    assert inc["status"] == "closed" and inc["start_frame"] == 15 and inc["end_frame"] == 35
    assert inc["track_ids"] == [1] and inc["evidence"]["weapon_hand"] == 0.6

    # clip: 1 s pre-roll (frames 5-14) + frames 15-35 (incl. the closing frame) + 1 s post-roll (36-45)
    clip = Path(inc["clip_path"])
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == 10 + 21 + 10
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == inc["clip_sha256"]
    assert Path(inc["snapshot_path"]).exists() and inc["snapshot_sha256"]

    detail = client.get(f"/api/incidents/{inc['id']}").json()
    assert [(e["scope"], e["to_level"]) for e in detail["events"]] == [
        ("track", "SUSPICIOUS_ACTIVITY"), ("scene", "SUSPICIOUS_ACTIVITY"), ("track", "WEAPON_RELATED_EVENT"),
        ("scene", "WEAPON_RELATED_EVENT"), ("track", "NORMAL"), ("scene", "NORMAL")]

    got = client.get(f"/api/incidents/{inc['id']}/clip", params={"actor": "reviewer"})
    assert got.status_code == 200 and got.headers["content-type"] == "video/mp4"
    ack = client.post(f"/api/incidents/{inc['id']}/ack", json={"actor": "reviewer", "note": "staged test"}).json()
    assert ack["ack_by"] == "reviewer" and ack["ack_note"] == "staged test"
    assert client.get("/api/incidents", params={"unacknowledged": True}).json() == []

    actions = [a["action"] for a in client.get("/api/audit").json()][::-1]
    for expected in ("run_created", "incident_opened", "incident_escalated", "incident_closed", "evidence_written",
                     "run_finished", "evidence_clip_accessed", "incident_acknowledged"):
        assert expected in actions, (expected, actions)
    assert actions.index("incident_opened") < actions.index("incident_closed") < actions.index("evidence_written")

    out = Path(run["out_dir"])
    for name in ("frames.json", "fusion_events.jsonl", "threat_timeline.csv", "summary.json", "incidents.json"):
        assert (out / name).exists(), name


def test_websocket_streams_run(client, video):
    with client.websocket_connect("/ws/events") as ws:
        assert ws.receive_json()["type"] == "hello"
        run_id = client.post("/api/runs", json={"source": str(video), "post_s": 0.5}).json()["id"]
        seen = []
        while True:
            m = ws.receive_json()
            seen.append(m["type"])
            if m["type"] in ("run_finished", "run_failed"):
                break
        assert m["type"] == "run_finished" and m["run_id"] == run_id
    for t in ("run_started", "frame", "level_change", "incident_opened", "incident_updated", "incident_closed"):
        assert t in seen, (t, seen)


def test_validation(client, video, tmp_path):
    outside = tmp_path / "elsewhere.avi"
    outside.write_bytes(video.read_bytes())
    assert client.post("/api/runs", json={"source": str(outside)}).status_code == 403
    assert client.post("/api/runs", json={"source": str(video.parent / "missing.avi")}).status_code == 422
    assert client.post("/api/runs", json={"source": str(video), "open_level": "NORMAL"}).status_code == 422
    assert client.post("/api/runs", json={"source": str(video), "weapon_model": str(video)}).status_code == 422
    assert client.get("/api/incidents/999").status_code == 404
    assert client.post("/api/runs/nope/stop").status_code == 404
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["levels"] == LEVELS

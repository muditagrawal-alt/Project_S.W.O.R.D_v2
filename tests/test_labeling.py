"""Interval-label review tool: derived pre-labels, revisions with provenance, validation, export, media access."""

import csv
import io

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fusion.labels import Interval, save_labels  # noqa: E402
from service.api import create_app  # noqa: E402

VID = "mendeley/Handgun/PAH1_C1_P1_V1_HB_1/video.mp4"


@pytest.fixture()
def client(tmp_path):
    root = tmp_path / "data"
    media = root / "mendeley_firearm/extracted/Gun_Action_Recognition_Dataset/Handgun/PAH1_C1_P1_V1_HB_1"
    media.mkdir(parents=True)
    (media / "video.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fake")
    save_labels([Interval(VID, 1.0, 4.0, "WEAPON_RELATED_EVENT", source="derived:mendeley_boxes", notes="PAH1"),
                 Interval("ucf/Robbery/Robbery048_x264.mp4", 2.0, 9.0, "SUSPICIOUS_ACTIVITY", source="derived:ucf_temporal",
                          notes="Robbery; NEEDS REVIEW")], root / "labels" / "x_derived.csv")
    save_labels([Interval("Cam7.json", 0, 1, "WEAPON_RELATED_EVENT")], root / "labels" / "usrt_derived_TEST_ONLY.csv")
    app = create_app(database_url=f"sqlite:///{tmp_path / 't.db'}", data_dir=str(tmp_path / "svc"),
                     label_data_root=str(root), make_system=lambda *a: None)
    with TestClient(app) as c:
        yield c


def test_review_flow(client):
    vids = {v["video"]: v for v in client.get("/api/label/videos").json()}
    assert set(vids) == {VID, "ucf/Robbery/Robbery048_x264.mp4"}          # TEST_ONLY labels are not offered
    assert vids[VID]["media"] and not vids["ucf/Robbery/Robbery048_x264.mp4"]["media"]
    assert vids["ucf/Robbery/Robbery048_x264.mp4"]["needs_review"]
    d = client.get("/api/label/intervals", params={"video": VID}).json()
    assert d["origin"] == "derived" and d["intervals"][0]["label"] == "WEAPON_RELATED_EVENT"

    body = {"video": VID, "annotator": "alice", "intervals": [
        {"start_s": 1.2, "end_s": 4.0, "label": "WEAPON_RELATED_EVENT", "verified_from_derived": True, "notes": "start corrected"},
        {"start_s": 4.0, "end_s": 6.0, "label": "NORMAL"}]}
    assert client.put("/api/label/intervals", json=body).json()["revision"] == 1
    body["annotator"] = "bob"
    assert client.put("/api/label/intervals", json=body).json()["revision"] == 2   # new revision, nothing overwritten
    d = client.get("/api/label/intervals", params={"video": VID}).json()
    assert d["origin"] == "reviewed" and d["revision"] == 2
    assert [i["source"] for i in d["intervals"]] == ["human-verified:bob", "human:bob"]

    rows = list(csv.DictReader(io.StringIO(client.get("/api/label/export").text)))
    assert [(r["video"], r["label"], r["source"]) for r in rows] == [(VID, "WEAPON_RELATED_EVENT", "human-verified:bob"),
                                                                     (VID, "NORMAL", "human:bob")]
    actions = [a["action"] for a in client.get("/api/audit").json()]
    assert actions.count("labels_saved") == 2


def test_validation_and_media(client):
    bad = {"video": VID, "annotator": "a", "intervals": [{"start_s": 5, "end_s": 1, "label": "NORMAL"}]}
    assert client.put("/api/label/intervals", json=bad).status_code == 422
    bad["intervals"] = [{"start_s": 0, "end_s": 1, "label": "DANGER"}]
    assert client.put("/api/label/intervals", json=bad).status_code == 422
    bad["intervals"] = []
    assert client.put("/api/label/intervals", json=bad).status_code == 422
    assert client.put("/api/label/intervals", json={"video": VID, "annotator": "", "intervals": [
        {"start_s": 0, "end_s": 1, "label": "NORMAL"}]}).status_code == 422
    r = client.get("/api/label/media", params={"video": VID})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert client.get("/api/label/media", params={"video": "mendeley/../../../../etc/passwd"}).status_code == 404
    assert client.get("/labeler").status_code == 200

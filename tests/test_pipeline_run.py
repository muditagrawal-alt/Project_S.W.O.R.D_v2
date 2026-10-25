"""pipeline.run: image-sequence sources keep their real timestamps (gaps) and serve as ground truth."""

import json

import cv2
import numpy as np

from fusion.rules import TemporalThreatEngine
from pipeline.run import ImageSequence, run_video


class EchoSystem:
    """Records the timestamps it is given; never raises an alarm."""
    sources = {"phase1": "echo"}

    def __init__(self):
        self.p5 = TemporalThreatEngine()
        self.ts = []

    def reset(self):
        self.ts = []

    def timing_summary(self):
        return {"total_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0}, "throughput_fps": 1000.0}

    def process(self, frame, idx, t):
        self.ts.append(t)
        return {"frame_idx": idx, "timestamp": t, "persons": [], "weapons": [], "weapon_detections": [], "actions": {},
                "events": [], "fusion_events": [], "threat": {"scene_level": "NORMAL", "tracks": {}}}


def make_sequence(tmp_path):
    numbers = [1, 2, 3, 6, 7]  # frames 4-5 missing -> 1.5 s gap at 2 fps
    frames = []
    for i, n in enumerate(numbers):
        name = f"f_{n}.jpg"
        cv2.imwrite(str(tmp_path / name), np.full((40, 60, 3), 30 * i, np.uint8))
        frames.append({"frame_idx": i, "frame_number": n, "timestamp": (n - 1) / 2.0, "image": name,
                       "boxes": [{"bbox": [1, 1, 5, 5], "label": "Handgun"}] if n in (6, 7) else []})
    path = tmp_path / "seq.json"
    path.write_text(json.dumps({"fps": 2.0, "width": 60, "height": 40, "frames": frames}))
    return path


def test_image_sequence_reader(tmp_path):
    seq = ImageSequence(str(make_sequence(tmp_path)))  # relative image paths resolve next to the JSON
    assert seq.isOpened() and (seq.w, seq.h) == (60, 40) and seq.fps == 2.0
    ts = []
    while True:
        ok, img = seq.read()
        if not ok:
            break
        ts.append(seq.last_t)
    assert ts == [0.0, 0.5, 1.0, 2.5, 3.0]


def test_run_video_uses_sequence_timestamps_and_gt(tmp_path):
    path = make_sequence(tmp_path)
    sys_ = EchoSystem()
    res = run_video(str(path), tmp_path / "out", system=sys_, gt=str(path))
    assert sys_.ts == [0.0, 0.5, 1.0, 2.5, 3.0]
    gt = res["evaluation"]["_ground_truth"]
    assert gt["events"] == 1 and gt["weapon_frames"] == 2 and gt["segments_s"] == [[2.5, 3.0]]
    assert res["evaluation"]["B4_rules_full"]["event_recall"] == 0.0

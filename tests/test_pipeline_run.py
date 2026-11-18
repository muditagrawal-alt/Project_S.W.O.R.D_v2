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


def make_video(tmp_path, n=30, fps=10.0):
    path = tmp_path / "clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for i in range(n):
        wr.write(np.full((48, 64, 3), (i * 8) % 255, np.uint8))
    wr.release()
    return path


def test_file_detects_on_every_kth_frame_but_shows_and_saves_every_frame(tmp_path):
    sys_, shown = EchoSystem(), []
    res = run_video(str(make_video(tmp_path)), tmp_path / "out", system=sys_, save_video=True, frame_stride=0,
                    on_display=lambda f: shown.append(f.shape))
    assert len(sys_.ts) == 15 and len(shown) == 30          # auto stride at 10 fps: 2
    assert res["summary"]["fps"] == 5.0
    cap = cv2.VideoCapture(str(tmp_path / "out" / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30 and round(cap.get(cv2.CAP_PROP_FPS)) == 10


def test_live_detection_runs_on_a_worker_thread_and_every_frame_is_shown(tmp_path):
    import threading
    import time

    class SlowSystem(EchoSystem):
        threads = []

        def process(self, frame, idx, t):
            self.threads.append(threading.current_thread().name)
            time.sleep(0.02)
            return super().process(frame, idx, t)

    sys_, shown, idxs = SlowSystem(), [], []
    t0 = time.monotonic()
    run_video(str(make_video(tmp_path)), tmp_path / "out", system=sys_, live=True, keep_frames=False,
              on_display=lambda f: shown.append(1), on_frame=lambda rec, f: idxs.append(rec["frame_idx"]))
    assert time.monotonic() - t0 >= 2.5                     # a file run live plays at its own rate (30 frames, 10 fps)
    assert len(shown) == 30 and 10 <= len(sys_.ts) <= 30     # detection keeps up roughly (20 ms each)
    assert idxs == list(range(len(idxs))) and sys_.ts == sorted(sys_.ts)
    assert all(n.startswith("sword-detect") for n in sys_.threads)

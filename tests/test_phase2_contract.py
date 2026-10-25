"""Phase 2 slot: the conformance checker accepts a valid tracker and reports contract violations."""

import cv2
import numpy as np

from tracking.contract import check_output, run_check


class GoodTracker:
    def reset(self):
        self.n = 0

    def update(self, frame, t):
        self.n += 1
        return [{"track_id": 1, "bbox": [10, 10, 40, 90], "score": 0.9, "trajectory": {"speed_heights_s": 0.2, "zone": None}},
                {"track_id": 2, "bbox": [50, 10, 70, 60], "score": 0.8, "trajectory": None}], []


def test_good_tracker_passes(tmp_path):
    path = str(tmp_path / "v.avi")
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 96))
    for _ in range(12):
        wr.write(np.zeros((96, 96, 3), np.uint8))
    wr.release()
    rep = run_check(GoodTracker(), path)
    assert rep["passed"] and rep["frames"] == 12 and rep["tracks"] == 2 and rep["track_length_median"] == 12


def test_violations_are_reported():
    shape = (100, 100, 3)
    bad = [{"track_id": "a", "bbox": [10, 10, 5, 50], "score": 0.9},
           {"track_id": "a", "bbox": [200, 200, 300, 300], "score": 0.9},
           {"bbox": [1, 2, 3, 4], "score": 0.5}]
    errs = check_output(bad, [{"type": "teleport"}], shape, 0.0)
    text = "\n".join(errs)
    for needle in ("track_id must be int", "non-positive size", "outside the frame", "missing 'track_id'",
                   "duplicate track_id", "bad zone event"):
        assert needle in text, needle

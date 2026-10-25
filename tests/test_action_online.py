"""Phase 4 streaming recognizer: windowing/stride rules and sync == async results (needs the PoseC3D checkpoint)."""

import numpy as np
import pytest

from action_recognition.posec3d import DEFAULT_CHECKPOINT

pytestmark = pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="PoseC3D checkpoint not downloaded")

FPS = 10
SHAPE = (480, 640, 3)
# COCO-17 standing pose (x, y) roughly 160 px tall
BASE = np.array([[0, 0], [-4, -4], [4, -4], [-8, -2], [8, -2], [-18, 20], [18, 20], [-24, 50], [24, 50], [-26, 78],
                 [26, 78], [-12, 82], [12, 82], [-14, 120], [14, 120], [-14, 158], [14, 158]], dtype=np.float32)


def people(t):
    out = []
    for tid, x0 in ((1, 200.0), (2, 320.0)):
        k = BASE + [x0 + 20 * t, 150.0]
        k[9:11, 1] -= 30 * np.sin(4 * t)  # moving wrists
        out.append({"track_id": tid, "keypoints": k.tolist(), "scores": [0.9] * 17})
    return out


def feed(rec, n_frames, wait_async=False):
    got = {}
    for i in range(n_frames):
        t = i / FPS
        for tid, r in rec.update(t, people(t), SHAPE).items():
            got[(tid, r["t"])] = r
        if wait_async and rec._future is not None:
            rec._future.result()  # deterministic: let each batch finish before the next frame
    return got


@pytest.fixture(scope="module")
def sync_results():
    from action_recognition.online import OnlineActionRecognizer

    rec = OnlineActionRecognizer(device="cpu", window_s=2.0, min_window_s=1.5, stride_s=1.0)
    return feed(rec, 40)


def test_windows_and_stride(sync_results):
    times = sorted({t for _, t in sync_results})
    assert times[0] == pytest.approx(1.5)  # first window once min_window_s of history exists
    assert np.allclose(np.diff(times), 1.0)  # then once per stride
    r = sync_results[(1, times[0])]
    assert r["partner_track_id"] == 2  # nearest co-occurring person used as NTU second skeleton
    assert len(r["probabilities_120"]) == 120 and abs(sum(r["probabilities_120"]) - 1) < 1e-2
    assert r["window_end"] - r["window_start"] <= 2.0 + 1e-6


def test_async_matches_sync(sync_results):
    from action_recognition.online import AsyncActionRecognizer

    rec = AsyncActionRecognizer(device="cpu", window_s=2.0, min_window_s=1.5, stride_s=1.0)
    got = feed(rec, 41, wait_async=True)  # one extra frame to collect the last batch
    rec.close()
    assert set(got) == set(sync_results)
    for key, r in got.items():
        assert r["top1_id"] == sync_results[key]["top1_id"]
        np.testing.assert_allclose(r["probabilities_120"], sync_results[key]["probabilities_120"], atol=1e-4)
        assert r["latency_s"] == pytest.approx(0.1)  # delivered one frame later


def test_track_without_enough_history_is_not_classified():
    from action_recognition.online import OnlineActionRecognizer

    rec = OnlineActionRecognizer(device="cpu", window_s=2.0, min_window_s=1.5, stride_s=1.0)
    assert feed(rec, 14) == {}  # 1.3 s of history < min_window_s


def test_linear_resampling_interpolates_visible_joints_only():
    from action_recognition.online import resample_linear

    ts = np.array([0.0, 1.0])
    k = np.zeros((2, 17, 2), np.float32)
    k[1] += 10.0
    s = np.ones((2, 17), np.float32)
    s[1, 0] = 0.0  # joint 0 invisible in the second frame -> nearest value, no interpolation
    kk, ss = resample_linear(ts, k, s, np.array([0.25, 0.5, 0.75]))
    np.testing.assert_allclose(kk[:, 1, 0], [2.5, 5.0, 7.5])
    assert kk[0, 0, 0] == 0.0 and kk[2, 0, 0] == 10.0 and ss[2, 0] == 0.0

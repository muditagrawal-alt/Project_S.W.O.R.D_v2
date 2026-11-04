"""Temporal rescoring (research/tools/phase1/temporal_rescore.py): persistence keeps scores, flickers drop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/tools/phase1"))
from temporal_rescore import rescore  # noqa: E402

BOX = [100, 100, 120, 140]           # 20 x 40 px; r = 1 -> 40 px search radius
NEAR = [110, 110, 130, 150]          # centre 14 px away
FAR = [400, 400, 420, 440]


def test_persistent_detection_keeps_score_and_flicker_drops():
    frames = [[(BOX, 0.8)], [(NEAR, 0.8)], [(BOX, 0.8)]]
    out = rescore(frames, k=1, r=1.0, lam=1.0)
    assert out[2] == [0.8] and out[1] == [0.8]
    assert out[0] == [0.4]                                    # nothing before the first frame counts as a miss
    flicker = rescore([[(FAR, 0.6)], [(BOX, 0.8)], [(FAR, 0.6)]], k=1, r=1.0, lam=1.0)
    assert flicker[1] == [0.4]                                # previous detection is too far away


def test_mixing_and_bidirectional_window():
    frames = [[], [(BOX, 0.8)], [(BOX, 0.4)]]
    assert rescore(frames, k=1, r=1.0, lam=0.5)[1] == [0.8 * 0.5 + 0.4 * 0.5]          # mean(0.8, 0) = 0.4
    both = rescore(frames, k=1, r=1.0, lam=1.0, causal=False)[1][0]
    assert abs(both - (0.8 + 0.0 + 0.4) / 3) < 1e-9                                       # one frame before, one after

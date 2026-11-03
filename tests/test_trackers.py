"""Phase 2 tracker wrappers: BoT-SORT keeps an identity across frames like ByteTrack and accepts the frame."""

import numpy as np

from tracking.tracker import BoTSORTWrapper, ByteTrackWrapper


def run(trk, with_img):
    frame = np.zeros((480, 640, 3), np.uint8)
    ids = []
    for i in range(8):
        box = np.array([[100 + 3 * i, 100, 150 + 3 * i, 250]], float)
        kw = {"img": frame} if with_img else {}
        out = trk.update(box, np.array([0.9]), **kw)
        ids += [t["track_id"] for t in out]
    return ids


def test_botsort_keeps_one_identity():
    ids = run(BoTSORTWrapper(track_buffer=5), with_img=True)
    assert ids and len(set(ids)) == 1


def test_bytetrack_keeps_one_identity():
    ids = run(ByteTrackWrapper(track_buffer=5), with_img=False)
    assert ids and len(set(ids)) == 1

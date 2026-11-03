"""Second-stage weapon verifier helpers (tracking/verifier.py): crops and score fusion."""

import numpy as np
import pytest

from tracking.verifier import crop_square, fuse_scores


def test_crop_is_square_resized_and_padded_at_the_edge():
    img = np.full((100, 200, 3), 255, np.uint8)
    c = crop_square(img, [0, 0, 20, 10], scale=2.0, size=64)          # half outside the frame
    assert c.shape == (64, 64, 3)
    assert c[0, 0].sum() == 0 and c[-1, -1].sum() > 0                  # zero padding top-left, image bottom-right


def test_fusion_modes():
    s, v = np.array([0.81, 0.5]), np.array([0.25, 1.0])
    assert np.allclose(fuse_scores(s, v, "geo"), [0.45, np.sqrt(0.5)])
    assert np.allclose(fuse_scores(s, v, "product"), [0.2025, 0.5])
    assert np.allclose(fuse_scores(s, v, "none"), s)
    with pytest.raises(ValueError):
        fuse_scores(s, v, "max")

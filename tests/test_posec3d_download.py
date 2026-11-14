"""PoseC3D checkpoint download on first use (action_recognition/posec3d.py ensure_checkpoint), offline."""

import hashlib
import urllib.request

import pytest

from action_recognition.posec3d import ensure_checkpoint

DATA = b"posec3d weights"
PREFIX = hashlib.sha256(DATA).hexdigest()[:8]


def fake_download(content):
    def retrieve(url, path):
        with open(path, "wb") as fp:
            fp.write(content)
    return retrieve


def test_download_is_checked_against_the_name(tmp_path, monkeypatch):
    target = tmp_path / f"slowonly-{PREFIX}.pth"
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_download(DATA))
    assert ensure_checkpoint(target) == target and target.read_bytes() == DATA
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a: pytest.fail("existing file must not be downloaded"))
    assert ensure_checkpoint(target) == target


def test_corrupt_download_is_rejected(tmp_path, monkeypatch):
    target = tmp_path / f"slowonly-{PREFIX}.pth"
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_download(b"corrupted"))
    with pytest.raises(RuntimeError):
        ensure_checkpoint(target)
    assert not target.exists() and not list(tmp_path.iterdir())

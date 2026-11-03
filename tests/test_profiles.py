"""System profiles (pipeline/profiles.py): built-in + file profiles, kwargs, fusion overrides, missing weights."""

import json

import pytest

import pipeline.profiles as P


def test_builtin_and_file_profiles(tmp_path, monkeypatch):
    f = tmp_path / "profiles.json"
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "det.pt").write_bytes(b"x")
    f.write_text(json.dumps({"default": "x", "profiles": {"x": {"weapon_model": "weights/det.pt", "weapon_imgsz": 1280,
                                                                 "weapon_zoom": ["hands"], "zoom_min_score": 0.3,
                                                                 "zoom_min_window": 96, "weapon_max_area": 0.125,
                                                                 "fusion": {"w_event": 0.6, "hold_s": 1.0}}}}))
    monkeypatch.setattr(P, "PROFILES_FILE", f)
    monkeypatch.setattr(P, "PROJECT_ROOT", tmp_path)
    assert set(P.list_profiles()) == {"original", "x"} and P.default_profile() == "x"
    kw = P.system_kwargs("x")
    assert kw["weapon_imgsz"] == 1280 and kw["weapon_zoom"] == ("hands",) and kw["zoom_min_score"] == 0.3
    assert kw["zoom_min_window"] == 96 and kw["weapon_max_area"] == 0.125
    cfg = P.fusion_config("x", fps=2.0)
    assert cfg.w_event == 0.6 and cfg.hold_s == 1.0 and cfg.grace_s > 1.0   # fps adaptation still applied
    with pytest.raises(FileNotFoundError):
        P.system_kwargs("original")   # weights/yolo26s_best.pt does not exist under tmp_path


def test_installer_maps_tuned_filters_out_of_fusion(tmp_path):
    """install_profile.py: detection filters from a tuning result go to the profile, rule values to its fusion."""
    import subprocess
    import sys

    tuning = tmp_path / "results.json"
    tuning.write_text(json.dumps({"selected": {"params": {"w_susp": 0.4, "w_event": 0.6, "persist_s": 1.0, "hold_s": 1.0,
                                                          "zoom_min_score": 0.3, "max_area_frac": 0.085, "zoom_min_window": 96}}}))
    (tmp_path / "det.pt").write_bytes(b"x")
    (tmp_path / "w").mkdir()
    subprocess.run([sys.executable, str(P.PROJECT_ROOT / "research/tools/integration/install_profile.py"), "--name", "t",
                    "--weights", str(tmp_path / "det.pt"), "--weights-name", "det.pt", "--zoom", "hands",
                    "--tuning", str(tuning), "--weights-dir", str(tmp_path / "w"), "--profiles-file", str(tmp_path / "p.json")],
                   check=True, capture_output=True)
    prof = json.loads((tmp_path / "p.json").read_text())["profiles"]["t"]
    assert prof["zoom_min_score"] == 0.3 and prof["weapon_max_area"] == 0.085 and prof["zoom_min_window"] == 96
    assert prof["fusion"] == {"w_susp": 0.4, "w_event": 0.6, "persist_s": 1.0, "hold_s": 1.0}
    assert "det.pt" in json.loads((tmp_path / "w" / "provenance.json").read_text())

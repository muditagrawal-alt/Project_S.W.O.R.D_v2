"""Command centre pages render without errors (Streamlit AppTest, no browser, no models loaded)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
VIEWS = Path(__file__).resolve().parent.parent / "command_center" / "views"


@pytest.mark.parametrize("page", ["overview", "results", "labeling", "monitor", "test_clip"])
def test_page_renders(page):
    at = st_testing.AppTest.from_file(str(VIEWS / f"{page}.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page has a title"


def test_results_page_lists_runs_and_switches():
    at = st_testing.AppTest.from_file(str(VIEWS / "results.py"), default_timeout=120).run()
    assert not at.exception
    if at.radio:   # local research runs exist: select another one and re-render
        options = at.radio[0].options
        if len(options) > 1:
            at.radio[0].set_value(options[1]).run()
            assert not at.exception


def test_test_clip_profile_switch_sets_options():
    from pipeline.profiles import list_profiles

    at = st_testing.AppTest.from_file(str(VIEWS / "test_clip.py"), default_timeout=120).run()
    assert not at.exception
    prof_box = next(s for s in at.selectbox if s.label.startswith("Profile"))
    for name in prof_box.options:
        prof_box.set_value(name).run()
        assert not at.exception, [e.value for e in at.exception]
        prof_box = next(s for s in at.selectbox if s.label.startswith("Profile"))
        size = next(s for s in at.select_slider if s.label == "Detector input size")
        assert size.value == list_profiles()[name]["weapon_imgsz"]


# ── the app runs from the right Python and reloads edited project modules ──
APP = VIEWS.parent / "app.py"


def test_app_puts_project_on_pythonpath_so_edits_reload(monkeypatch):
    """Streamlit only reloads edited modules next to the script or on PYTHONPATH; without the project root there, an
    edit to pipeline/run.py left the running app with the old module (ImportError: cannot import name ...)."""
    import os
    monkeypatch.delenv("PYTHONPATH", raising=False)
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception and not at.error
    assert str(VIEWS.parent.parent) in os.environ["PYTHONPATH"].split(os.pathsep)


def test_app_explains_missing_packages_instead_of_crashing(monkeypatch):
    monkeypatch.syspath_prepend(str(VIEWS.parent))
    import lib
    monkeypatch.setitem(lib.REQUIRED_PACKAGES, "sword_no_such_module", "sword-no-such-package")
    at = st_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.error and "sword-no-such-package" in at.error[0].value and "venv" in at.error[0].value


def test_service_starts_with_the_venv_python_when_this_one_lacks_packages(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(VIEWS.parent))
    import lib
    venv_python = tmp_path / "python"
    venv_python.touch()
    calls = []
    monkeypatch.setattr(lib, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(lib, "VENV_PYTHON", venv_python)
    monkeypatch.setattr(lib.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(lib, "missing_packages", lambda: ["sqlalchemy"])
    lib.start_service_background(8123)
    monkeypatch.setattr(lib, "missing_packages", lambda: [])
    lib.start_service_background(8123)
    import sys
    assert calls[0][:3] == [str(venv_python), "-m", "service"] and calls[1][0] == sys.executable

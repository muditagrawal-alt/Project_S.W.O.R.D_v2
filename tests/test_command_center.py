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

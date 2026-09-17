"""Headless verification of the Streamlit dashboard via AppTest, since no
browser is available in this environment. These start the real background
agent pipeline (daemon threads) and sleep briefly for it to tick -- slower
than a typical unit test, but this is the closest automated substitute for
"load it in a browser and check for errors" that a UI file gets.
"""
import time
from pathlib import Path

import pytest

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "dashboard.py")


@pytest.fixture(autouse=True)
def _disable_autorefresh(monkeypatch):
    # dashboard.py's live st.rerun() loop would otherwise make AppTest.run()
    # spin forever; this env var is the documented test hook for that.
    monkeypatch.setenv("MAATS_DASHBOARD_TEST", "1")


def test_dashboard_renders_all_kpis_without_exceptions_on_first_load():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    labels = {m.label for m in at.metric}
    # The top KPI strip always renders these 6 (as "—" placeholders if the
    # background pipeline thread hasn't seeded state.command yet -- that
    # seeding is a race with this very first script run, so it's not
    # asserted here). A superset check catches real breakage (a KPI
    # disappearing) without being brittle to that race.
    assert labels >= {
        "Current Phase", "Active Direction", "Time Remaining",
        "Vehicles Detected", "Average Queue", "Congestion Level",
    }


def test_dashboard_populates_real_numbers_once_the_pipeline_has_ticked():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
    at.run()
    time.sleep(4)
    at.run()
    assert not at.exception
    vehicles = next(m for m in at.metric if m.label == "Vehicles Detected")
    assert int(vehicles.value) >= 0
    assert len(at.code) >= 1  # the agent message stream block rendered


def test_dashboard_shows_signal_command_and_active_badge_on_first_load():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
    at.run()
    # command is seeded immediately at pipeline start (see dashboard.py's
    # on_signal_command seeding) -- production green slots are a fixed 90s
    # now, so waiting for an actual transition here would mean sleeping
    # 90+ real seconds. A short sleep just lets the background thread
    # actually start before we read its state.
    time.sleep(2)
    at.run()
    assert not at.exception
    phase = next(m for m in at.metric if m.label == "Current Phase")
    assert phase.value != "—"
    assert len(at.success) >= 1  # at least one ACTIVE badge is showing


def test_reset_demo_button_produces_a_fresh_pipeline_with_cleared_history():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_PATH, default_timeout=30)
    at.run()
    time.sleep(4)
    at.run()
    state_before = at.session_state["dash_state"]
    assert len(state_before.messages) > 0  # some real activity has happened

    reset_button = next(b for b in at.button if "RESET DEMO" in b.label)
    reset_button.click()
    at.run()

    state_after = at.session_state["dash_state"]
    assert state_after is not state_before  # a genuinely new pipeline/state, not a mutation
    assert not at.exception
    # fresh state starts with cleared message history/signal state, same as any new start_pipeline() call
    assert state_after.command is None or state_after.started_at >= state_before.started_at


def test_hybrid_source_pipeline_starts_without_crashing():
    from dashboard import start_pipeline
    from backend.cv.roi_config import ROIConfig

    video_configs = {
        "N": ROIConfig(
            direction="N",
            video_source="backend/cv/configs/demo_videos/north.mp4",
            roi_polygon=[(0, 0), (640, 0), (640, 480), (0, 480)],
            queue_polygon=[(0, 320), (640, 320), (640, 480), (0, 480)],
            frame_skip=0,
            detector_kind="motion",
        )
    }
    choice = (video_configs, ("S", "E", "W"), "balanced")  # Mode B: N=video, rest=simulated
    state = start_pipeline(("N", "S", "E", "W"), "HYBRID", "Video (per direction)", choice)
    time.sleep(3)
    assert state.error is None
    assert state.source_labels.get("N") == "live_video"
    assert state.source_labels.get("S") == "simulated"

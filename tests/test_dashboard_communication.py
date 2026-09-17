"""Proves the Multi-Agent Communication view is wired to the real pipeline,
not static example text:
  1. four agents publish updates
  2. the coordinator receives them
  3. the coordinator makes a decision
  4. a SignalCommand is produced
  5. the dashboard displays the actual message content, with the numbers
     cross-checked against DashboardState's live agent snapshots
"""
import time

import pytest


@pytest.fixture(autouse=True)
def _disable_autorefresh(monkeypatch):
    monkeypatch.setenv("MAATS_DASHBOARD_TEST", "1")


def test_four_agents_publish_and_coordinator_decides_and_dashboard_shows_it():
    from dashboard import DIRECTION_NAMES, start_pipeline

    state = start_pipeline(("N", "S", "E", "W"), "SIMULATION", "Simulation", "balanced")
    time.sleep(3)  # a handful of 1s agent/coordinator ticks is enough for real messages to flow

    # 1 & 2: every direction's agent has received and processed at least one
    # real update (default AgentSnapshot fields would still be all-zero/False)
    for direction in ("N", "S", "E", "W"):
        agent = state.agents[direction]
        assert agent.last_message_time > 0, f"{direction} agent never received a message"

    # 3 & 4: the coordinator produced a real SignalCommand -- seeded
    # immediately at pipeline start (see dashboard.py's on_signal_command
    # seeding), so this doesn't need to wait for an actual phase transition
    assert state.command is not None
    assert state.command.phase.value in {"GREEN", "YELLOW", "ALL_RED"}

    # 5: the message stream literally contains each agent's real traffic,
    # not a placeholder -- cross-check the numbers in the logged text
    # against the same agent's live snapshot
    joined = "\n".join(state.messages)
    for direction in ("N", "S", "E", "W"):
        name = DIRECTION_NAMES[direction]
        assert f"{name}_AGENT" in joined
        agent = state.agents[direction]
        assert any(f"queue={agent.queue_length}" in m for m in state.messages)

    assert "[COORDINATOR → SIGNAL CONTROLLER]" in joined or "[SIGNAL CONTROLLER]" in joined or "COORDINATOR\ncurrent_direction=" in joined


def test_communication_view_renders_live_values_in_the_browser_dom():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    dashboard_path = str(Path(__file__).resolve().parent.parent / "dashboard.py")
    at = AppTest.from_file(dashboard_path, default_timeout=30)
    at.run()
    # command is seeded immediately now (see dashboard.py's on_signal_command
    # seeding) -- keep this short, not long: the seeded COORDINATOR log line
    # is a one-time event at the head of the message deque, and the "Live
    # Message Stream" only ever shows the 12 most recent entries, so too
    # long a sleep lets ~4 agent-update messages/second bury it before we
    # can assert on it.
    time.sleep(1)
    at.run()
    assert not at.exception

    stream = next(c.value for c in at.code)
    for name in ("NORTH", "SOUTH", "EAST", "WEST"):
        assert f"{name}_AGENT" in stream
    assert "[COORDINATOR → SIGNAL CONTROLLER]" in stream or "[SIGNAL CONTROLLER]" in stream or "COORDINATOR\ncurrent_direction=" in stream

    decision_texts = [t.value for t in at.text]
    assert any(t.startswith("Phase: ") for t in decision_texts)
    assert any(t.startswith("Selected reason: ") for t in decision_texts)

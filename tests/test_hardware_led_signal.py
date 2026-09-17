"""ESP32/LED hardware mapping: the same single-green invariant applied to
the physical output layer, plus the mandatory fail-safe guard before any
hardware write."""
from __future__ import annotations

import logging

from backend.hardware.led_signal import (
    assert_single_green,
    assert_single_green_state,
    compute_led_state,
    fail_safe_all_red,
    send_signal_state,
)
from backend.traffic_engine.signal_fsm import Phase, SignalCommand

DIRECTIONS = ("N", "E", "S", "W")


def _command(phase, active_direction):
    return SignalCommand(
        phase=phase, active_direction=active_direction, started_at=0.0,
        duration=90.0, time_remaining=45.0, starvation_override=False,
    )


def test_compute_led_state_sets_exactly_one_green_during_green_phase():
    state = compute_led_state(DIRECTIONS, _command(Phase.GREEN, "N"))
    assert state["N"] == "GREEN"
    for direction in ("E", "S", "W"):
        assert state[direction] == "RED"
    assert sum(1 for v in state.values() if v == "GREEN") == 1


def test_compute_led_state_for_every_possible_active_direction():
    for direction in DIRECTIONS:
        state = compute_led_state(DIRECTIONS, _command(Phase.GREEN, direction))
        greens = [d for d, light in state.items() if light == "GREEN"]
        assert greens == [direction]


def test_compute_led_state_yellow_belongs_to_only_the_active_direction():
    state = compute_led_state(DIRECTIONS, _command(Phase.YELLOW, "E"))
    assert state["E"] == "YELLOW"
    for direction in ("N", "S", "W"):
        assert state[direction] == "RED"


def test_compute_led_state_all_red_means_every_direction_red():
    state = compute_led_state(DIRECTIONS, _command(Phase.ALL_RED, "W"))
    assert all(light == "RED" for light in state.values())


def test_assert_single_green_state_passes_through_a_safe_state():
    safe_state = {"N": "GREEN", "E": "RED", "S": "RED", "W": "RED"}
    assert assert_single_green_state(safe_state) == safe_state


def test_assert_single_green_state_fails_safe_on_two_greens():
    unsafe_state = {"N": "GREEN", "E": "RED", "S": "GREEN", "W": "RED"}
    result = assert_single_green_state(unsafe_state)
    assert result == fail_safe_all_red(("N", "E", "S", "W"))
    assert all(light == "RED" for light in result.values())


def test_assert_single_green_state_logs_the_violation(caplog):
    unsafe_state = {"N": "GREEN", "E": "GREEN", "S": "RED", "W": "RED"}
    with caplog.at_level(logging.ERROR, logger="maats.hardware"):
        assert_single_green_state(unsafe_state)
    assert any("SAFETY VIOLATION" in record.message for record in caplog.records)


def test_assert_single_green_state_never_raises_on_a_violation():
    unsafe_state = {"N": "GREEN", "E": "GREEN", "S": "GREEN", "W": "GREEN"}
    result = assert_single_green_state(unsafe_state)  # must not raise
    assert all(light == "RED" for light in result.values())


def test_send_signal_state_always_routes_through_the_safety_gate():
    written = []
    send_signal_state(DIRECTIONS, _command(Phase.GREEN, "S"), write=written.append)
    assert len(written) == 1
    assert written[0]["S"] == "GREEN"
    assert sum(1 for v in written[0].values() if v == "GREEN") == 1


def test_send_signal_state_never_writes_two_greens_across_every_direction():
    written = []
    for direction in DIRECTIONS:
        send_signal_state(DIRECTIONS, _command(Phase.GREEN, direction), write=written.append)
    for state in written:
        assert sum(1 for v in state.values() if v == "GREEN") <= 1


# -- assert_single_green(): the named invariant monitor required at BOTH
# boundaries (dashboard rendering and hardware output) --

def test_assert_single_green_is_the_composite_of_compute_and_gate():
    for direction in DIRECTIONS:
        command = _command(Phase.GREEN, direction)
        assert assert_single_green(DIRECTIONS, command) == assert_single_green_state(
            compute_led_state(DIRECTIONS, command)
        )


def test_assert_single_green_fails_safe_on_a_command_whose_active_direction_is_unknown():
    # A hand-built or corrupted SignalCommand pointing at a direction that
    # isn't in the intersection's own topology -- compute_led_state's own
    # "if active_direction in state" guard already keeps this to all-RED,
    # and assert_single_green must agree (never crash, never fabricate a
    # green for an unrecognized direction).
    command = _command(Phase.GREEN, "NE")  # not a real direction
    result = assert_single_green(DIRECTIONS, command)
    assert all(light == "RED" for light in result.values())

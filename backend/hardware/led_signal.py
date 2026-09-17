"""ESP32/LED hardware mapping -- the layer between a SignalCommand and
whatever actually drives the physical LEDs (serial/GPIO/wifi to the board).

No transport (serial port, GPIO pins, HTTP call to the ESP32) is wired in
here: there is no physical board in this development environment to test
against, and fabricating a serial protocol nobody can verify would be
worse than not having one. What this module DOES own is the safety
contract every transport must go through before a single bit reaches the
hardware -- that part is fully real and fully tested.

Real hardware integration plugs in by writing one function,
`write: Dict[str, str] -> None` (e.g. wrapping a serial.write() call), and
passing it to send_signal_state() below. That function is the ONLY
sanctioned way to reach hardware from this module, and it always routes
through assert_single_green_state() first -- so it is not possible to send
an unsafe two-green command by skipping the guard, only by bypassing this
module entirely.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

from ..traffic_engine.signal_fsm import Phase, SignalCommand

logger = logging.getLogger("maats.hardware")


def compute_led_state(directions: Tuple[str, ...], command: SignalCommand) -> Dict[str, str]:
    """Pure mapping: SignalCommand -> {direction: "RED"|"YELLOW"|"GREEN"}.
    Exactly one direction can come out non-RED: command carries a single
    active_direction field, so there is no code path here that could ever
    set two directions to GREEN -- the old NS+EW paired-axis bug is
    structurally unrepresentable at this layer, same as in signal_fsm.py
    and dashboard.direction_light()."""
    state = {d: "RED" for d in directions}
    if command.active_direction in state:
        if command.phase == Phase.GREEN:
            state[command.active_direction] = "GREEN"
        elif command.phase == Phase.YELLOW:
            state[command.active_direction] = "YELLOW"
        # ALL_RED: every direction stays RED, already the default above.
    return state


def fail_safe_all_red(directions: Tuple[str, ...]) -> Dict[str, str]:
    return {d: "RED" for d in directions}


def assert_single_green_state(led_state: Dict[str, str]) -> Dict[str, str]:
    """The final safety gate before ANY hardware write. Returns the state
    that is actually safe to send: led_state unchanged if at most one
    GREEN is present, or a fail-safe ALL-RED state (with the violation
    logged) otherwise.

    Deliberately never raises: a hardware control loop must always get
    back a state it can apply immediately, not an exception that leaves
    the previous (possibly also stale) LED state on the board."""
    green_directions = [d for d, light in led_state.items() if light == "GREEN"]
    if len(green_directions) > 1:
        logger.error(
            "HARDWARE SAFETY VIOLATION: %d directions requested GREEN simultaneously (%s) -- "
            "failing safe to ALL RED",
            len(green_directions), green_directions,
        )
        return fail_safe_all_red(tuple(led_state.keys()))
    return led_state


def assert_single_green(directions: Tuple[str, ...], command: SignalCommand) -> Dict[str, str]:
    """THE named invariant monitor: the one function every consumer of a
    SignalCommand -- dashboard rendering or hardware output -- must call at
    its final boundary before turning that command into pixels or volts.
    Composes compute_led_state() + assert_single_green_state() into a
    single call so there is exactly one place this check can be forgotten,
    not two. Returns a state guaranteed to have at most one GREEN; on a
    violation it fails safe to ALL RED and logs the exact cause (which
    direction(s) were wrongly GREEN) via assert_single_green_state()."""
    return assert_single_green_state(compute_led_state(directions, command))


def send_signal_state(
    directions: Tuple[str, ...],
    command: SignalCommand,
    write: Callable[[Dict[str, str]], None],
) -> Dict[str, str]:
    """The one sanctioned entry point for driving real hardware: computes
    the LED state, runs it through the safety gate, THEN calls `write` --
    so a caller cannot reach hardware without going through the guard.
    `write` is whatever the actual transport is (serial.write, GPIO calls,
    an HTTP request to the ESP32, ...); this function is transport-agnostic
    on purpose (see module docstring)."""
    state = assert_single_green(directions, command)
    write(state)
    return state

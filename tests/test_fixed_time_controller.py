"""FixedTimeController: the deterministic "no video / no traffic data"
fallback. Must rotate NORTH -> EAST -> SOUTH -> WEST -> repeat, each on a
constant duration, never influenced by traffic, never stalling."""
from __future__ import annotations

from backend.traffic_engine.fixed_time_controller import FixedTimeController
from backend.traffic_engine.signal_fsm import Phase

LEGAL_EDGES = {
    Phase.GREEN: {Phase.YELLOW},
    Phase.YELLOW: {Phase.ALL_RED},
    Phase.ALL_RED: {Phase.GREEN},
}


def test_fixed_time_rotates_north_east_south_west_regardless_of_traffic():
    now = 0.0
    controller = FixedTimeController(green_duration=5.0, clock=lambda: now)

    greens_seen = [controller.fsm.active_direction]
    previous = controller.fsm.phase
    for _ in range(400):
        now += 0.25
        command = controller.tick(now)
        if command is not None:
            assert command.phase in LEGAL_EDGES[previous]
            previous = command.phase
            if command.phase == Phase.GREEN:
                greens_seen.append(command.active_direction)
        if len(greens_seen) >= 5:
            break

    assert greens_seen == ["N", "E", "S", "W", "N"]


def test_fixed_time_green_duration_is_constant_never_adaptive():
    now = 0.0
    controller = FixedTimeController(green_duration=15.0, clock=lambda: now)
    for _ in range(200):
        now += 0.5
        command = controller.tick(now)
        if command is not None and command.phase == Phase.GREEN:
            assert command.duration == 15.0


def test_fixed_time_never_stalls_over_many_cycles():
    now = 0.0
    controller = FixedTimeController(green_duration=2.0, clock=lambda: now)
    transitions = 0
    for _ in range(4000):
        now += 0.1
        if controller.tick(now) is not None:
            transitions += 1
    # 3 transitions per direction slot (GREEN->YELLOW->ALL_RED->GREEN), many
    # full cycles over this span -- zero would mean the controller stalled
    assert transitions > 30


def test_configurable_sequence_and_default_north_east_south_west():
    default_controller = FixedTimeController(green_duration=1.0, clock=lambda: 0.0)
    assert default_controller.fsm.directions == ("N", "E", "S", "W")

    custom_controller = FixedTimeController(directions=("S", "N"), green_duration=1.0, clock=lambda: 0.0)
    assert custom_controller.fsm.active_direction == "S"


def test_no_video_fallback_gives_every_direction_exactly_ninety_seconds_by_default():
    # The TRUE, unconfigured default -- no injected short test duration --
    # matching "NORTH=90, EAST=90, SOUTH=90, WEST=90" for the no-video/
    # no-metrics case specifically.
    now = 0.0
    controller = FixedTimeController(clock=lambda: now)  # no green_duration override at all
    assert controller.fsm.command(now).duration == 90.0

    seen_durations = []
    for direction in ("N", "E", "S", "W"):
        assert controller.fsm.active_direction == direction
        assert controller.fsm.command(now).duration == 90.0
        seen_durations.append(controller.fsm.command(now).duration)
        for _ in range(3):  # GREEN -> YELLOW -> ALL_RED -> GREEN(next)
            now += controller.fsm.time_remaining(now) + 0.001
            controller.tick(now)
    assert seen_durations == [90.0, 90.0, 90.0, 90.0]

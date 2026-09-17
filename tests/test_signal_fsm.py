"""Strict single-direction signal control invariants.

These are the tests that matter most for this rewrite: the old NS_GREEN/
EW_GREEN paired-axis model made N+S (or E+W) green *together* by design.
Every test below is either a direct regression test for that exact bug
(now structurally impossible, since SignalCommand only ever carries one
active_direction) or a check on the fixed-duration/safety-transition rules
that replace it.
"""
from __future__ import annotations

import pytest

from backend.traffic_engine.signal_fsm import (
    ALL_RED_DURATION,
    DEFAULT_DIRECTION_SEQUENCE,
    GREEN_DURATION_SECONDS,
    MIN_GREEN_SECONDS,
    YELLOW_DURATION,
    Phase,
    SignalFSM,
)

ALL_DIRECTIONS = ("N", "E", "S", "W")

LEGAL_EDGES = {
    Phase.GREEN: {Phase.YELLOW},
    Phase.YELLOW: {Phase.ALL_RED},
    Phase.ALL_RED: {Phase.GREEN},
}


def _green_count(fsm: SignalFSM, now: float) -> int:
    return sum(1 for d in fsm.directions if fsm.light_for(d, now) == "GREEN")


def test_green_duration_constant_is_ninety_seconds():
    assert GREEN_DURATION_SECONDS == 90.0


def test_default_sequence_is_north_east_south_west():
    assert DEFAULT_DIRECTION_SEQUENCE == ("N", "E", "S", "W")


def test_advance_before_duration_elapsed_is_a_noop():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    assert fsm.advance(5.0) is False
    assert fsm.phase == Phase.GREEN
    assert fsm.active_direction == "N"


def test_yellow_and_all_red_durations_are_fixed_regardless_of_green_duration():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.advance(10.0)
    assert fsm.phase == Phase.YELLOW
    assert fsm.time_remaining(10.0) == YELLOW_DURATION

    fsm.advance(10.0 + YELLOW_DURATION)
    assert fsm.phase == Phase.ALL_RED
    assert fsm.time_remaining(10.0 + YELLOW_DURATION) == ALL_RED_DURATION


# -- CORE RULE: exactly one direction GREEN at any instant, ever --

def test_exactly_one_direction_green_during_a_green_phase():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    assert fsm.phase == Phase.GREEN
    assert _green_count(fsm, 0.0) == 1
    for direction in ALL_DIRECTIONS:
        if direction == "N":
            assert fsm.light_for(direction, 0.0) == "GREEN"
        else:
            assert fsm.light_for(direction, 0.0) == "RED"


def test_at_most_one_green_at_every_instant_across_a_long_run():
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=5.0, clock=lambda: now)
    for _ in range(2000):
        now += 0.25
        fsm.advance(now)
        green_count = _green_count(fsm, now)
        assert green_count <= 1, f"more than one green at t={now}: phase={fsm.phase}, active={fsm.active_direction}"


def test_every_possible_active_direction_never_produces_a_second_green():
    # regression test for the exact dashboard bug: N+S (or E+W) both green.
    # Drive each direction to active in turn and confirm the other three
    # are red for the entire duration of its slot, for every direction.
    for start_direction in ALL_DIRECTIONS:
        now = 0.0
        fsm = SignalFSM(initial_direction=start_direction, green_duration=5.0, clock=lambda: now)
        for _ in range(20):
            now += 0.25
            fsm.advance(now)
            for direction in ALL_DIRECTIONS:
                if direction == fsm.active_direction:
                    continue
                assert fsm.light_for(direction, now) == "RED", (
                    f"{direction} was not RED while {fsm.active_direction} was active "
                    f"(started rotation at {start_direction})"
                )


def test_north_and_south_are_never_both_green_the_old_bug_this_replaces():
    """Direct regression test for the reported bug: under the old NS_GREEN/
    EW_GREEN model, N and S were ALWAYS the same light (paired on one
    axis), so N green implied S green too. Confirm that's gone."""
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=5.0, clock=lambda: now)
    for _ in range(40):
        now += 0.25
        fsm.advance(now)
        n_green = fsm.light_for("N", now) == "GREEN"
        s_green = fsm.light_for("S", now) == "GREEN"
        e_green = fsm.light_for("E", now) == "GREEN"
        w_green = fsm.light_for("W", now) == "GREEN"
        assert not (n_green and s_green), f"N+S both green at t={now} -- the exact old bug"
        assert not (e_green and w_green), f"E+W both green at t={now} -- the exact old bug"


# -- Fixed 90s (or injected test duration) green, never dynamic --

def test_green_duration_is_never_extended_by_request_next_mid_phase():
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: now)
    duration_before = fsm.command(now).duration
    fsm.request_next("E", starvation=False)  # a traffic score "just changed" mid-green
    now = 5.0
    assert fsm.command(now).duration == duration_before  # unchanged -- not interrupted, not extended
    assert fsm.phase == Phase.GREEN
    assert fsm.active_direction == "N"  # still N -- the pending request hasn't applied yet


def test_current_green_cannot_be_interrupted_by_a_newly_calculated_duration():
    # The realistic case: the coordinator recomputes E's traffic mid-phase
    # and sends a brand new CALCULATED duration for E's eventual slot. N's
    # currently-running green (with its own already-fixed duration) must be
    # completely unaffected -- active direction, phase, AND duration.
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=37.0, clock=lambda: now)
    fsm.request_next("E", green_duration=82.0, starvation=False)  # E's newly calculated slot
    now = 20.0  # well into N's green, traffic keeps changing
    fsm.request_next("E", green_duration=15.0, starvation=False)  # E's traffic dropped since
    command = fsm.command(now)
    assert command.active_direction == "N"
    assert command.phase == Phase.GREEN
    assert command.duration == 37.0  # N's own slot, completely untouched
    assert command.time_remaining == pytest.approx(17.0)


def test_timer_counts_down_monotonically_to_zero():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    remainders = [fsm.time_remaining(now) for now in [0.0, 2.5, 5.0, 7.5, 9.9, 10.0]]
    assert remainders == pytest.approx([10.0, 7.5, 5.0, 2.5, 0.1, 0.0])
    for earlier, later in zip(remainders, remainders[1:]):
        assert later <= earlier  # monotonically non-increasing, never jumps up


def test_next_direction_does_not_become_green_before_the_timer_reaches_zero():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", starvation=False)
    assert fsm.advance(9.99) is False
    assert fsm.active_direction == "N"
    assert fsm.phase == Phase.GREEN


# -- Safe transitions: GREEN -> YELLOW -> ALL_RED -> GREEN(next), never a shortcut --

def test_full_round_trip_follows_the_legal_sequence_only():
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: now)
    fsm.request_next("E", starvation=False)

    transitions = []
    previous = fsm.phase
    requested_return = False
    for _ in range(400):
        now += 0.5
        if fsm.advance(now):
            transitions.append((previous, fsm.phase))
            previous = fsm.phase
            if fsm.phase == Phase.GREEN and fsm.active_direction == "E" and not requested_return:
                fsm.request_next("N", starvation=False)
                requested_return = True
            if fsm.phase == Phase.GREEN and fsm.active_direction == "N" and requested_return:
                break

    for src, dst in transitions:
        assert dst in LEGAL_EDGES[src], f"illegal transition {src} -> {dst}"
    assert transitions[0] == (Phase.GREEN, Phase.YELLOW)
    assert (Phase.ALL_RED, Phase.GREEN) in transitions


def test_yellow_belongs_to_only_the_active_direction():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.advance(10.0)
    assert fsm.phase == Phase.YELLOW
    assert fsm.light_for("N", 10.0) == "YELLOW"
    for direction in ("E", "S", "W"):
        assert fsm.light_for(direction, 10.0) == "RED"


def test_all_red_means_every_direction_is_red():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.advance(10.0)
    fsm.advance(10.0 + YELLOW_DURATION)
    assert fsm.phase == Phase.ALL_RED
    for direction in ALL_DIRECTIONS:
        assert fsm.light_for(direction, 10.0 + YELLOW_DURATION) == "RED"


def test_pending_direction_only_applied_at_all_red_exit():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", starvation=False)

    fsm.advance(10.0)
    assert fsm.phase == Phase.YELLOW
    assert fsm.active_direction == "N"  # request_next never touched the active direction early
    fsm.advance(10.0 + YELLOW_DURATION)
    assert fsm.phase == Phase.ALL_RED

    fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)
    assert fsm.phase == Phase.GREEN
    assert fsm.active_direction == "E"
    assert fsm.command().duration == 10.0  # still the fixed green_duration, not something request_next set


# -- Default no-video/no-traffic behavior: deterministic round robin, never stalls --

def test_default_rotation_is_strict_north_east_south_west_when_nobody_requests_next():
    now = 0.0
    fsm = SignalFSM(green_duration=1.0, clock=lambda: now)  # short duration, fast test
    seen = [fsm.active_direction]
    for _ in range(4):
        for _ in range(3):  # GREEN -> YELLOW -> ALL_RED -> GREEN(next) takes 3 advances
            now += fsm.time_remaining(now) + 0.001
            fsm.advance(now)
        seen.append(fsm.active_direction)
    assert seen == ["N", "E", "S", "W", "N"]  # full cycle, then repeats


def test_default_rotation_never_stalls_never_skips_a_direction():
    now = 0.0
    fsm = SignalFSM(green_duration=1.0, clock=lambda: now)
    visited = [fsm.active_direction]  # the initial GREEN period, entered at construction, not via advance()
    for _ in range(4 * 3 * 5):  # 5 full cycles worth of phase advances
        now += fsm.time_remaining(now) + 0.001  # jump straight to the next phase boundary
        if fsm.advance(now) and fsm.phase == Phase.GREEN:
            visited.append(fsm.active_direction)
    # every direction appears, and always in the fixed order, cycling
    assert len(visited) >= 5
    for i, direction in enumerate(visited):
        assert direction == DEFAULT_DIRECTION_SEQUENCE[i % 4]


def test_configurable_sequence_is_respected():
    fsm = SignalFSM(directions=("E", "W", "N", "S"), green_duration=1.0, clock=lambda: 0.0)
    assert fsm.active_direction == "E"


# -- Starvation influences WHICH direction, never the duration --

def test_starvation_override_changes_direction_not_duration():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("W", starvation=True)
    fsm.advance(10.0)
    fsm.advance(10.0 + YELLOW_DURATION)
    fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)

    command = fsm.command()
    assert command.starvation_override is True
    assert command.active_direction == "W"
    assert command.duration == 10.0  # exactly the configured green_duration, starvation didn't change it


# -- Adaptive duration: request_next() can specify a per-slot green_duration --

def test_request_next_applies_a_calculated_duration_for_the_next_slot():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", green_duration=47.0, starvation=False)
    fsm.advance(10.0)  # -> YELLOW(N)
    fsm.advance(10.0 + YELLOW_DURATION)  # -> ALL_RED
    fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)  # -> GREEN(E)
    command = fsm.command()
    assert command.active_direction == "E"
    assert command.duration == 47.0  # the calculated value, not the FSM's own 10.0 default


def test_request_next_clamps_a_duration_below_min_green():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", green_duration=2.0, starvation=False)  # below MIN_GREEN_SECONDS
    fsm.advance(10.0)
    fsm.advance(10.0 + YELLOW_DURATION)
    fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)
    assert fsm.command().duration == MIN_GREEN_SECONDS


def test_request_next_clamps_a_duration_above_max_green():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", green_duration=500.0, starvation=False)  # way above GREEN_DURATION_SECONDS
    fsm.advance(10.0)
    fsm.advance(10.0 + YELLOW_DURATION)
    fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)
    assert fsm.command().duration == GREEN_DURATION_SECONDS


def test_a_per_slot_duration_override_does_not_persist_to_the_following_slot():
    # request_next() with an explicit duration only affects the NEXT slot --
    # if nobody calls it again, the FSM falls back to its own configured
    # default (the no-video/no-traffic-data deterministic behavior), not
    # the previous slot's calculated value.
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", green_duration=82.0, starvation=False)
    fsm.advance(10.0)
    fsm.advance(10.0 + YELLOW_DURATION)
    fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)  # -> GREEN(E), duration 82
    assert fsm.command().duration == 82.0

    # nobody calls request_next this time -- E's slot ends via default rotation
    now = 10.0 + YELLOW_DURATION + ALL_RED_DURATION
    now += 82.0
    fsm.advance(now)  # -> YELLOW(E)
    now += YELLOW_DURATION
    fsm.advance(now)  # -> ALL_RED
    now += ALL_RED_DURATION
    fsm.advance(now)  # -> GREEN(next), no override this time
    assert fsm.command().duration == 10.0  # back to the FSM's own configured default

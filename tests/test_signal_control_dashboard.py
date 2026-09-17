"""Behavioral tests for the dedicated Signal Control dashboard view.

These exercise the real chain (DirectionalAgent/CoordinatorAgent -> SignalFSM
-> SignalCommand -> dashboard.direction_light/starvation_banner_text), the
same objects dashboard.py renders from. Nothing here fakes a countdown or
invents a light color independently of the FSM.

test_4/test_4b below are direct regression tests for the reported bug: the
old NS_GREEN/EW_GREEN model made North+South (and East+West) the SAME
light, always, by construction -- direction_light("N", cmd) and
direction_light("S", cmd) were guaranteed equal. That guarantee is gone;
these tests assert the opposite is now true.
"""
from __future__ import annotations

from backend.agents.coordinator_agent import CoordinatorAgent, DIRECTIONS
from backend.agents.message_bus import MessageBus
from backend.traffic_engine.metrics import PriorityUpdate
from backend.traffic_engine.signal_fsm import ALL_RED_DURATION, YELLOW_DURATION, Phase, SignalFSM

from dashboard import AgentSnapshot, direction_light, starvation_banner_text

LEGAL_EDGES = {
    Phase.GREEN: {Phase.YELLOW},
    Phase.YELLOW: {Phase.ALL_RED},
    Phase.ALL_RED: {Phase.GREEN},
}


def _update(direction, vehicle_count, priority_score, time_since_last_green=5.0, is_starved=False):
    return PriorityUpdate(
        direction=direction, priority_score=priority_score, vehicle_count=vehicle_count,
        queue_length=vehicle_count, avg_wait_time=1.0, arrival_rate=1.0,
        time_since_last_green=time_since_last_green, is_starved=is_starved, timestamp=0.0,
    )


def test_1_coordinator_picks_east_as_highest_priority_direction():
    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0)
    coordinator.ingest(_update("N", vehicle_count=1, priority_score=1.0))
    coordinator.ingest(_update("S", vehicle_count=1, priority_score=1.0))
    coordinator.ingest(_update("E", vehicle_count=25, priority_score=5.0))
    coordinator.ingest(_update("W", vehicle_count=1, priority_score=1.0))

    direction, green_duration, starvation = coordinator.decide()
    assert direction == "E" and starvation is False
    assert green_duration == 90.0  # E's heavy traffic (25 vehicles/queue) clamps to MAX_GREEN


def test_2_east_winning_gets_a_green_duration_calculated_from_its_own_traffic():
    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0, initial_direction="N", green_duration=10.0)
    coordinator.ingest(_update("N", vehicle_count=1, priority_score=1.0))
    coordinator.ingest(_update("S", vehicle_count=1, priority_score=1.0))
    coordinator.ingest(_update("E", vehicle_count=25, priority_score=5.0))  # heavy East traffic
    coordinator.ingest(_update("W", vehicle_count=1, priority_score=1.0))

    coordinator.fsm.advance(10.0)  # GREEN(N) -> YELLOW(N)
    coordinator.fsm.advance(10.0 + YELLOW_DURATION)  # -> ALL_RED
    coordinator.fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)  # -> GREEN(E)
    command = coordinator.fsm.command()
    assert command.active_direction == "E"
    # E's own heavy traffic (25 vehicles, queue 25) clamps to MAX_GREEN=90 --
    # calculated from E's real metrics, not N's initial fixed 10.0
    assert command.duration == 90.0


def test_3_starvation_forces_service_and_the_dashboard_shows_the_override_banner():
    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0, initial_direction="N", green_duration=10.0)
    # South is starved (waited well past STARVATION_THRESHOLD) despite the
    # lowest priority score of the four -- it must still win.
    coordinator.ingest(_update("N", vehicle_count=5, priority_score=5.0, time_since_last_green=10.0))
    coordinator.ingest(_update("S", vehicle_count=1, priority_score=0.1, time_since_last_green=130.0, is_starved=True))
    coordinator.ingest(_update("E", vehicle_count=3, priority_score=3.0, time_since_last_green=10.0))
    coordinator.ingest(_update("W", vehicle_count=2, priority_score=2.0, time_since_last_green=10.0))

    direction, green_duration, starvation = coordinator.decide()
    assert direction == "S" and starvation is True

    coordinator.fsm.request_next(direction, green_duration=green_duration, starvation=starvation)
    coordinator.fsm.advance(10.0)  # GREEN(N) -> YELLOW(N)
    coordinator.fsm.advance(10.0 + YELLOW_DURATION)  # -> ALL_RED
    coordinator.fsm.advance(10.0 + YELLOW_DURATION + ALL_RED_DURATION)  # -> GREEN(S)
    command = coordinator.fsm.command()

    assert command.starvation_override is True
    assert command.active_direction == "S"
    # starvation affects WHICH direction wins, never the duration calculation
    assert command.duration == green_duration

    agents = {"S": AgentSnapshot("S", time_since_last_green=0.0)}
    banner = starvation_banner_text(command, agents)
    assert banner is not None
    assert "STARVATION OVERRIDE" in banner
    assert "SOUTH" in banner


def test_4_north_and_south_are_never_the_same_light_the_exact_dashboard_bug():
    """Direct regression test for the reported bug: North and South showed
    the SAME light color (both green together) because the old FSM paired
    them on one axis. Drive a real FSM through several full cycles and
    confirm N and S (and E and W) are independent -- never both GREEN."""
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: now)
    fsm.request_next("E", starvation=False)

    for _ in range(400):
        now += 0.5
        fsm.advance(now)
        command = fsm.command(now)
        lights = {d: direction_light(d, command) for d in DIRECTIONS}
        green_count = sum(1 for light in lights.values() if light == "GREEN")
        assert green_count <= 1, f"more than one green at t={now}: {lights}"
        assert not (lights["N"] == "GREEN" and lights["S"] == "GREEN"), f"N+S both green at t={now}"
        assert not (lights["E"] == "GREEN" and lights["W"] == "GREEN"), f"E+W both green at t={now}"


def test_4b_every_direction_shows_green_alone_never_paired_with_its_old_axis_partner():
    # Old model: N's axis partner was S, E's was W. Confirm each direction's
    # green slot leaves its former "partner" red the entire time.
    old_partner = {"N": "S", "S": "N", "E": "W", "W": "E"}
    for direction in DIRECTIONS:
        now = 0.0
        fsm = SignalFSM(initial_direction=direction, green_duration=5.0, clock=lambda: now)
        for _ in range(20):
            now += 0.25
            fsm.advance(now)
            if fsm.active_direction == direction and fsm.phase == Phase.GREEN:
                command = fsm.command(now)
                partner = old_partner[direction]
                assert direction_light(direction, command) == "GREEN"
                assert direction_light(partner, command) == "RED", (
                    f"{partner} was GREEN while {direction} was GREEN -- the old paired-axis bug"
                )


def test_5_timer_reaching_zero_produces_only_a_legal_phase_transition():
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: 0.0)
    fsm.request_next("E", starvation=False)

    before = fsm.command(9.9)
    assert before.time_remaining > 0
    assert fsm.advance(9.9) is False  # timer hasn't actually reached zero yet

    at_zero = fsm.command(10.0)
    assert at_zero.time_remaining == 0.0
    previous_phase = fsm.phase
    transitioned = fsm.advance(10.0)
    assert transitioned is True
    assert fsm.phase in LEGAL_EDGES[previous_phase]


def test_6_dashboard_direction_light_matches_fsm_light_for_every_direction():
    now = 0.0
    fsm = SignalFSM(initial_direction="N", green_duration=10.0, clock=lambda: now)
    for _ in range(80):
        now += 0.5
        fsm.advance(now)
        command = fsm.command(now)
        for direction in DIRECTIONS:
            assert direction_light(direction, command) == fsm.light_for(direction, now)


def test_7_dashboard_direction_light_fails_safe_on_a_corrupted_command():
    # direction_light() is routed through backend.hardware.led_signal.
    # assert_single_green() -- the same named invariant monitor the
    # ESP32/hardware output path uses. A hand-built SignalCommand pointing
    # at an unrecognized "active_direction" must render as all-RED, never
    # crash the dashboard and never show a green for a direction that
    # doesn't exist in this intersection's topology.
    from backend.traffic_engine.signal_fsm import SignalCommand

    corrupted = SignalCommand(
        phase=Phase.GREEN, active_direction="NE", started_at=0.0,
        duration=90.0, time_remaining=45.0, starvation_override=False,
    )
    for direction in DIRECTIONS:
        assert direction_light(direction, corrupted) == "RED"

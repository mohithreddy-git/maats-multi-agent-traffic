import pytest

from backend.agents.coordinator_agent import CoordinatorAgent, DIRECTIONS
from backend.agents.directional_agent import DirectionalAgent
from backend.agents.message_bus import MessageBus
from backend.traffic_engine.metrics import PriorityUpdate, TrafficMetrics
from backend.traffic_engine.signal_fsm import Phase, SignalCommand


def make_metrics_batch(**overrides):
    base = {
        d: TrafficMetrics(direction=d, vehicle_count=2, queue_length=2, avg_wait_time=5.0, arrival_rate=2.0, timestamp=0.0)
        for d in DIRECTIONS
    }
    base.update(overrides)
    return base


def test_directional_agent_computes_density_against_full_batch():
    bus = MessageBus()
    agent = DirectionalAgent("N", bus, clock=lambda: 0.0, peak_override=False)

    batch = make_metrics_batch(
        N=TrafficMetrics(direction="N", vehicle_count=5, queue_length=5, avg_wait_time=10.0, arrival_rate=3.0, timestamp=0.0),
    )
    # total vehicle_count across N,S,E,W = 5 + 2 + 2 + 2 = 11
    update = agent.process_metrics(batch, now=0.0)
    assert update.direction == "N"
    assert update.vehicle_count == 5
    assert update.priority_score > 0


def test_directional_agent_resets_time_since_last_green_on_matching_direction_signal():
    bus = MessageBus()
    now = [0.0]
    agent = DirectionalAgent("N", bus, clock=lambda: now[0], peak_override=False)
    now[0] = 200.0

    assert agent.time_since_last_green == pytest.approx(200.0)

    command = SignalCommand(
        phase=Phase.GREEN, active_direction="N", started_at=200.0, duration=15.0,
        time_remaining=15.0, starvation_override=False,
    )
    agent._on_signal_command(command, now=now[0])
    assert agent.time_since_last_green == pytest.approx(0.0)


def test_directional_agent_ignores_signal_command_for_a_different_direction():
    bus = MessageBus()
    now = [0.0]
    agent = DirectionalAgent("N", bus, clock=lambda: now[0], peak_override=False)
    now[0] = 200.0

    command = SignalCommand(
        phase=Phase.GREEN, active_direction="E", started_at=200.0, duration=15.0,
        time_remaining=15.0, starvation_override=False,
    )
    agent._on_signal_command(command, now=now[0])
    assert agent.time_since_last_green == pytest.approx(200.0)  # unchanged


def test_directional_agent_ignores_yellow_for_its_own_direction():
    # only an actual GREEN resets the wait clock -- yellow is a transition
    # out of green, not a new green slot starting
    bus = MessageBus()
    now = [0.0]
    agent = DirectionalAgent("N", bus, clock=lambda: now[0], peak_override=False)
    now[0] = 200.0

    command = SignalCommand(
        phase=Phase.YELLOW, active_direction="N", started_at=197.0, duration=3.0,
        time_remaining=3.0, starvation_override=False,
    )
    agent._on_signal_command(command, now=now[0])
    assert agent.time_since_last_green == pytest.approx(200.0)  # unchanged


def _make_update(direction, priority_score, time_since_last_green, is_starved, vehicle_count=1, queue_length=1, density=0.0):
    return PriorityUpdate(
        direction=direction,
        priority_score=priority_score,
        vehicle_count=vehicle_count,
        queue_length=queue_length,
        avg_wait_time=1.0,
        arrival_rate=1.0,
        time_since_last_green=time_since_last_green,
        is_starved=is_starved,
        timestamp=0.0,
        density=density,
    )


def test_coordinator_forces_starved_direction_regardless_of_score():
    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0)

    coordinator.ingest(_make_update("N", priority_score=5.0, time_since_last_green=10.0, is_starved=False))
    coordinator.ingest(_make_update("S", priority_score=0.1, time_since_last_green=130.0, is_starved=True))
    coordinator.ingest(_make_update("E", priority_score=3.0, time_since_last_green=10.0, is_starved=False))
    coordinator.ingest(_make_update("W", priority_score=2.0, time_since_last_green=10.0, is_starved=False))

    direction, green_duration, starvation = coordinator.decide()
    assert direction == "S"
    assert starvation is True
    assert isinstance(green_duration, float)


def test_coordinator_picks_highest_score_when_no_starvation():
    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0)

    coordinator.ingest(_make_update("N", priority_score=1.0, time_since_last_green=5.0, is_starved=False))
    coordinator.ingest(_make_update("S", priority_score=1.2, time_since_last_green=5.0, is_starved=False))
    coordinator.ingest(_make_update("E", priority_score=3.5, time_since_last_green=5.0, is_starved=False))
    coordinator.ingest(_make_update("W", priority_score=2.0, time_since_last_green=5.0, is_starved=False))

    direction, green_duration, starvation = coordinator.decide()
    assert direction == "E"
    assert starvation is False


def test_coordinator_decide_returns_a_duration_bounded_by_min_and_max():
    # decide() must produce a (direction, green_duration, starvation) triple
    # -- the duration is real (computed from the winner's own metrics via
    # compute_green_time()), always within [MIN_GREEN, MAX_GREEN].
    from backend.traffic_engine.scoring import MAX_GREEN, MIN_GREEN

    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0)
    coordinator.ingest(_make_update("N", priority_score=5.0, time_since_last_green=5.0, is_starved=False))
    coordinator.ingest(_make_update("S", priority_score=1.0, time_since_last_green=5.0, is_starved=False))
    coordinator.ingest(_make_update("E", priority_score=1.0, time_since_last_green=5.0, is_starved=False))
    coordinator.ingest(_make_update("W", priority_score=1.0, time_since_last_green=5.0, is_starved=False))

    result = coordinator.decide()
    assert len(result) == 3  # (direction, green_duration, starvation)
    direction, green_duration, starvation = result
    assert isinstance(direction, str)
    assert isinstance(green_duration, float)
    assert isinstance(starvation, bool)
    assert MIN_GREEN <= green_duration <= MAX_GREEN


def test_coordinator_ingest_drives_the_fsm_to_the_decided_direction_with_a_calculated_duration():
    bus = MessageBus()
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0, initial_direction="N", green_duration=10.0)
    coordinator.ingest(_make_update("N", priority_score=1.0, time_since_last_green=5.0, is_starved=False, vehicle_count=1, queue_length=1, density=0.0))
    coordinator.ingest(_make_update("S", priority_score=1.0, time_since_last_green=5.0, is_starved=False))
    # E wins on priority AND has heavy traffic of its own -- its green
    # duration must come from ITS metrics (16 vehicles, queue 5, density 0.8)
    coordinator.ingest(_make_update("E", priority_score=9.0, time_since_last_green=5.0, is_starved=False, vehicle_count=16, queue_length=5, density=0.8))
    coordinator.ingest(_make_update("W", priority_score=1.0, time_since_last_green=5.0, is_starved=False))

    # N's slot must run its full initial duration regardless of E's higher score
    assert coordinator.fsm.advance(10.0) is True  # GREEN(N) -> YELLOW(N)
    assert coordinator.fsm.active_direction == "N"
    coordinator.fsm.advance(13.0)  # -> ALL_RED
    coordinator.fsm.advance(14.0)  # -> GREEN(next)
    command = coordinator.fsm.command(14.0)
    assert command.active_direction == "E"  # E's higher priority won the NEXT slot
    # compute_green_time(vehicle_count=16, queue_length=5, density=0.8) == 10 + 32 + 15 + 16 == 73
    assert command.duration == 73.0  # calculated from E's OWN traffic, not the initial fixed 10.0

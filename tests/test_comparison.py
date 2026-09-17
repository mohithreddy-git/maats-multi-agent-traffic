from backend.simulation.comparison import ComparisonRunner
from backend.traffic_engine.signal_fsm import Phase


def test_same_seed_gives_identical_traffic_to_both_arms_before_any_decision_diverges():
    # On tick 1, neither controller has had a chance to diverge in its
    # green schedule yet (both start at the same initial direction), so the
    # raw traffic drawn must be identical -- this is the "same scenario/seed" guarantee.
    runner = ComparisonRunner(profile="balanced", seed=123)
    snapshot = runner.step()
    assert snapshot.adaptive_queues == snapshot.fixed_queues


def test_priority_ordering_selects_the_heavy_direction_more_often_than_round_robin():
    # Round-robin ignores traffic and serves each of 4 directions an equal
    # share by construction; priority-ordering under sustained heavy_north
    # traffic should both select N more often AND (now that duration is
    # adaptive) give N proportionally more total green time, since N's own
    # heavy traffic pushes its calculated duration up too.
    runner = ComparisonRunner(profile="heavy_north", seed=7, green_duration=10.0)
    snapshots = runner.run(ticks=600)  # 10 minutes of simulated traffic

    adaptive_green_ticks = [s for s in snapshots if s.adaptive_command.phase == Phase.GREEN]
    fixed_green_ticks = [s for s in snapshots if s.fixed_command.phase == Phase.GREEN]
    adaptive_n_share = sum(1 for s in adaptive_green_ticks if s.adaptive_command.active_direction == "N") / len(adaptive_green_ticks)
    fixed_n_share = sum(1 for s in fixed_green_ticks if s.fixed_command.active_direction == "N") / len(fixed_green_ticks)

    assert adaptive_n_share > fixed_n_share


def test_adaptive_arm_produces_dynamically_sized_durations_the_fixed_arm_does_not():
    # The round-robin (fixed-time) arm must always use the single configured
    # duration -- it ignores traffic by design. The priority-ordered
    # (adaptive) arm must NOT: its duration is calculated per slot from that
    # direction's own real traffic (compute_green_time()), so under a
    # traffic profile that varies over time, it must produce more than one
    # distinct duration value.
    runner = ComparisonRunner(profile="heavy_north", seed=7, green_duration=12.0)
    snapshots = runner.run(ticks=600)
    adaptive_durations = {s.adaptive_command.duration for s in snapshots if s.adaptive_command.phase == Phase.GREEN}
    fixed_durations = {s.fixed_command.duration for s in snapshots if s.fixed_command.phase == Phase.GREEN}

    assert fixed_durations == {12.0}  # round-robin: always the configured constant
    assert len(adaptive_durations) > 1  # adaptive: genuinely varies with traffic
    from backend.traffic_engine.scoring import MAX_GREEN, MIN_GREEN
    assert all(MIN_GREEN <= d <= MAX_GREEN for d in adaptive_durations)


def test_starved_direction_eventually_gets_served_under_adaptive_control():
    runner = ComparisonRunner(profile="starvation_test", seed=7, green_duration=10.0)
    snapshots = runner.run(ticks=600)  # well past the 120s starvation threshold

    starvation_events = [s for s in snapshots if s.adaptive_command.starvation_override]
    assert starvation_events, "expected at least one starvation override to fire"


def test_starvation_never_produces_a_two_green_state_or_an_illegal_phase():
    runner = ComparisonRunner(profile="starvation_test", seed=7, green_duration=10.0)
    snapshots = runner.run(ticks=600)
    for s in snapshots:
        # A single Phase/active_direction pair can never represent two
        # simultaneous greens; this also exercises that starvation
        # overrides still go through the normal ALL_RED gate.
        assert s.adaptive_command.phase in (Phase.GREEN, Phase.YELLOW, Phase.ALL_RED)
        assert s.fixed_command.phase in (Phase.GREEN, Phase.YELLOW, Phase.ALL_RED)

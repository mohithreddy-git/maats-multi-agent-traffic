import pytest

from backend.traffic_engine.scoring import MAX_GREEN, MIN_GREEN, compute_green_time, compute_priority_score


def test_compute_priority_score_matches_hand_calculation():
    # queue_norm=0.5, wait_norm=0.5, arrival_norm=0.5, density_norm=0.25
    # score = 0.30*0.5 + 0.30*0.5 + 0.15*0.5 + 0.15*0.25 + 1.0 - 0.15*0
    score = compute_priority_score(
        queue_length=10,
        time_since_last_green=45,
        arrival_rate=5,
        vehicle_count=5,
        total_vehicle_count=20,
        is_peak=False,
    )
    assert score == pytest.approx(1.4125)


def test_peak_multiplier_adds_fixed_amount():
    base = compute_priority_score(
        queue_length=0, time_since_last_green=0, arrival_rate=0,
        vehicle_count=1, total_vehicle_count=4, is_peak=False,
    )
    peak = compute_priority_score(
        queue_length=0, time_since_last_green=0, arrival_rate=0,
        vehicle_count=1, total_vehicle_count=4, is_peak=True,
    )
    assert peak - base == pytest.approx(0.2)


def test_normalized_terms_cap_at_one():
    saturated = compute_priority_score(
        queue_length=1000, time_since_last_green=1000, arrival_rate=1000,
        vehicle_count=4, total_vehicle_count=4, is_peak=False,
    )
    more_saturated = compute_priority_score(
        queue_length=2000, time_since_last_green=2000, arrival_rate=2000,
        vehicle_count=4, total_vehicle_count=4, is_peak=False,
    )
    assert saturated == more_saturated


def test_downstream_penalty_reduces_score():
    without_penalty = compute_priority_score(
        queue_length=10, time_since_last_green=10, arrival_rate=2,
        vehicle_count=2, total_vehicle_count=8, is_peak=False, downstream_penalty=0.0,
    )
    with_penalty = compute_priority_score(
        queue_length=10, time_since_last_green=10, arrival_rate=2,
        vehicle_count=2, total_vehicle_count=8, is_peak=False, downstream_penalty=1.0,
    )
    assert with_penalty == pytest.approx(without_penalty - 0.15)


# -- compute_green_time(): the adaptive green-duration calculation --

def test_green_time_zero_traffic_gets_the_minimum():
    assert compute_green_time(vehicle_count=0, queue_length=0, density=0.0) == MIN_GREEN


def test_green_time_low_traffic_is_short():
    duration = compute_green_time(vehicle_count=2, queue_length=1, density=0.1)
    assert MIN_GREEN < duration < 30.0


def test_green_time_medium_traffic_is_medium():
    duration = compute_green_time(vehicle_count=9, queue_length=3, density=0.45)
    assert 30.0 <= duration < 60.0


def test_green_time_heavy_traffic_is_long_but_not_maxed():
    duration = compute_green_time(vehicle_count=16, queue_length=5, density=0.8)
    assert 60.0 <= duration < MAX_GREEN


def test_green_time_extreme_traffic_reaches_the_maximum():
    duration = compute_green_time(vehicle_count=25, queue_length=12, density=1.0)
    assert duration == MAX_GREEN


def test_green_time_never_goes_below_min_green_even_for_negative_inputs():
    # defensive: malformed/negative metrics must still fail safe to a valid duration
    assert compute_green_time(vehicle_count=-5, queue_length=-5, density=-1.0) == MIN_GREEN


def test_green_time_never_exceeds_max_green_for_absurd_inputs():
    assert compute_green_time(vehicle_count=10_000, queue_length=10_000, density=100.0) == MAX_GREEN


def test_green_time_is_a_pure_deterministic_function_of_its_inputs():
    args = dict(vehicle_count=14, queue_length=9, density=0.7)
    results = {compute_green_time(**args) for _ in range(20)}
    assert len(results) == 1  # same inputs -> exactly the same output, every time


def test_more_traffic_always_yields_a_longer_or_equal_green_time():
    low = compute_green_time(vehicle_count=1, queue_length=0, density=0.05)
    medium = compute_green_time(vehicle_count=9, queue_length=3, density=0.45)
    heavy = compute_green_time(vehicle_count=16, queue_length=5, density=0.8)
    assert low < medium < heavy

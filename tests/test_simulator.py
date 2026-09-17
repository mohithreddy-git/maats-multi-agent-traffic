import random

import pytest

from backend.simulation.simulator import PROFILE_NAMES, Simulator


def _run(profile: str, ticks: int, seed: int = 1) -> dict:
    sim = Simulator(profile=profile, rng=random.Random(seed))
    totals = {d: 0 for d in sim.directions}
    for _ in range(ticks):
        batch = sim.tick()
        for d, m in batch.items():
            totals[d] += m.queue_length
    return totals


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_all_profiles_produce_valid_metrics_for_all_four_directions(profile):
    sim = Simulator(profile=profile, rng=random.Random(1))
    batch = sim.tick()
    assert set(batch.keys()) == {"N", "S", "E", "W"}
    for m in batch.values():
        assert m.vehicle_count >= 0
        assert m.queue_length >= 0
        assert m.arrival_rate >= 0


def test_heavy_north_accumulates_more_queue_on_north_than_other_directions():
    totals = _run("heavy_north", ticks=200)
    assert totals["N"] > totals["S"]
    assert totals["N"] > totals["E"]
    assert totals["N"] > totals["W"]


def test_starvation_test_profile_keeps_the_ns_axis_far_lighter_than_ew():
    # N and S share an axis (both get green together), so a meaningful
    # starvation scenario has to starve the whole axis, not one direction.
    totals = _run("starvation_test", ticks=200)
    assert totals["N"] < totals["E"]
    assert totals["N"] < totals["W"]
    assert totals["S"] < totals["E"]
    assert totals["S"] < totals["W"]


def test_balanced_profile_keeps_directions_roughly_comparable():
    totals = _run("balanced", ticks=300)
    values = list(totals.values())
    assert max(values) < 3 * min(values) + 10  # no direction wildly dominates


def test_green_directions_drain_queue_faster_than_red():
    seed = 7
    sim_green = Simulator(profile="heavy_north", rng=random.Random(seed))
    sim_red = Simulator(profile="heavy_north", rng=random.Random(seed))

    for _ in range(100):
        sim_green.tick(green_directions=frozenset({"N"}))
        sim_red.tick(green_directions=frozenset())

    assert sim_green._queue_length["N"] < sim_red._queue_length["N"]


def test_same_seed_and_green_schedule_produces_identical_output():
    seed = 99
    sim_a = Simulator(profile="balanced", rng=random.Random(seed))
    sim_b = Simulator(profile="balanced", rng=random.Random(seed))

    for _ in range(50):
        batch_a = sim_a.tick(green_directions=frozenset({"N", "S"}))
        batch_b = sim_b.tick(green_directions=frozenset({"N", "S"}))
        for d in sim_a.directions:
            assert batch_a[d].vehicle_count == batch_b[d].vehicle_count
            assert batch_a[d].queue_length == batch_b[d].queue_length


def test_set_profile_rejects_unknown_name():
    sim = Simulator()
    with pytest.raises(ValueError):
        sim.set_profile("does_not_exist")

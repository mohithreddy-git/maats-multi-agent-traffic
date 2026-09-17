"""Pure priority-scoring and green-time functions.

Kept dependency-free and side-effect-free so they can be unit tested without
any agent/asyncio machinery.
"""
from __future__ import annotations

from datetime import datetime

from .signal_fsm import GREEN_DURATION_SECONDS as MAX_GREEN
from .signal_fsm import MIN_GREEN_SECONDS as MIN_GREEN

# Priority weights (spec-fixed)
W_QUEUE = 0.30
W_WAIT = 0.30
W_ARRIVAL = 0.15
W_DENSITY = 0.15
W_DOWNSTREAM = 0.15

# Normalization caps
QUEUE_CAP = 20.0
WAIT_CAP = 90.0
ARRIVAL_CAP = 10.0  # vehicles/min; demo default, tune per intersection

STARVATION_THRESHOLD = 120.0

PEAK_MULTIPLIER = 1.2
OFF_PEAK_MULTIPLIER = 1.0
DEFAULT_PEAK_WINDOWS = ((7, 9), (17, 19))  # hour-of-day ranges, demo defaults


def is_peak_hour(hour: int, windows=DEFAULT_PEAK_WINDOWS) -> bool:
    return any(start <= hour < end for start, end in windows)


def is_peak_hour_now(windows=DEFAULT_PEAK_WINDOWS) -> bool:
    return is_peak_hour(datetime.now().hour, windows)


def compute_priority_score(
    *,
    queue_length: float,
    time_since_last_green: float,
    arrival_rate: float,
    vehicle_count: int,
    total_vehicle_count: int,
    is_peak: bool,
    downstream_penalty: float = 0.0,
    arrival_cap: float = ARRIVAL_CAP,
) -> float:
    """P_i = w1*QueueNorm + w2*WaitNorm + w3*ArrivalNorm + w4*DensityNorm
    + PeakMultiplier - w5*DownstreamPenalty (additive multiplier, per spec)."""
    queue_norm = min(queue_length / QUEUE_CAP, 1.0)
    wait_norm = min(time_since_last_green / WAIT_CAP, 1.0)
    arrival_norm = min(arrival_rate / arrival_cap, 1.0) if arrival_cap > 0 else 0.0
    density_norm = vehicle_count / max(1, total_vehicle_count)
    peak_multiplier = PEAK_MULTIPLIER if is_peak else OFF_PEAK_MULTIPLIER

    return (
        W_QUEUE * queue_norm
        + W_WAIT * wait_norm
        + W_ARRIVAL * arrival_norm
        + W_DENSITY * density_norm
        + peak_multiplier
        - W_DOWNSTREAM * downstream_penalty
    )


def is_starved(time_since_last_green: float) -> bool:
    return time_since_last_green > STARVATION_THRESHOLD


# -- Adaptive green-time calculation --
#
# MAX_GREEN (90s) is the hard upper bound owned by SignalFSM itself (the
# safety-critical module) -- imported here, not duplicated, so there is
# exactly one number for "the maximum a green phase can ever run."
#
# Weights below are tuned against the real traffic ranges actually measured
# on this project's demo/validation videos (real 1280x720 uploaded traffic
# footage and the synthetic light/medium/heavy clips), not guessed:
#   - empty/light traffic:  vehicle_count ~0-2,  queue ~0-1, density ~0.0-0.1
#   - medium traffic:       vehicle_count ~4-9,  queue ~2-3, density ~0.2-0.45
#   - heavy real traffic:   vehicle_count ~14-19, queue ~3-5, density ~0.7-0.9
# BASE_GREEN equals MIN_GREEN so a direction with zero traffic gets exactly
# the configured minimum, not something arbitrarily higher.
BASE_GREEN = 10.0
VEHICLE_WEIGHT = 2.0
QUEUE_WEIGHT = 3.0
DENSITY_WEIGHT = 20.0  # density is normalized 0..1, so this weight sits on a different scale than the count-based ones


def compute_green_time(*, vehicle_count: float, queue_length: float, density: float) -> float:
    """Deterministic, explainable adaptive green duration for the direction
    that just won the next slot -- reads ONLY that direction's own current
    TrafficMetrics-derived numbers (never fabricated, never averaged across
    directions). Called exactly once per slot, when the direction is
    selected -- see CoordinatorAgent.decide().

        green_duration = clamp(
            BASE_GREEN
            + VEHICLE_WEIGHT * vehicle_count
            + QUEUE_WEIGHT * queue_length
            + DENSITY_WEIGHT * density,
            MIN_GREEN, MAX_GREEN,
        )
    """
    raw = BASE_GREEN + VEHICLE_WEIGHT * vehicle_count + QUEUE_WEIGHT * queue_length + DENSITY_WEIGHT * density
    return max(MIN_GREEN, min(MAX_GREEN, raw))

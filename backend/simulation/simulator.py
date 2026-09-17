"""Configurable traffic-profile simulator.

Produces the same TrafficMetrics batch shape the YOLO adapter will produce
later, so the two sources are interchangeable for every agent downstream --
neither DirectionalAgent nor CoordinatorAgent has any notion of "simulated"
vs "real" data.

Arrival/departure physics: each direction gets a per-tick arrival
probability (the "profile"), and a fixed departure probability that is
higher while that direction currently has a green light -- modelling
saturation flow (roughly one vehicle clearing every ~2s of green, a
standard single-lane traffic-engineering assumption) vs a small permitted-
turn leak while red.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Callable, Dict, FrozenSet, Optional, Tuple

from ..agents.message_bus import MessageBus
from ..traffic_engine.metrics import TrafficMetrics

DIRECTIONS: Tuple[str, ...] = ("N", "S", "E", "W")

# Per-tick arrival probability per direction. -1.0 is a sentinel meaning
# "redraw a random probability every tick" (used by the "random" profile).
PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {"N": 0.3, "S": 0.3, "E": 0.3, "W": 0.3},
    "heavy_north": {"N": 0.85, "S": 0.2, "E": 0.2, "W": 0.2},
    "heavy_south": {"N": 0.2, "S": 0.85, "E": 0.2, "W": 0.2},
    "heavy_east": {"N": 0.2, "S": 0.2, "E": 0.85, "W": 0.2},
    "heavy_west": {"N": 0.2, "S": 0.2, "E": 0.2, "W": 0.85},
    "random": {"N": -1.0, "S": -1.0, "E": -1.0, "W": -1.0},
    "starvation_test": {"N": 0.05, "S": 0.05, "E": 0.8, "W": 0.8},
}
PROFILE_NAMES: Tuple[str, ...] = tuple(PROFILES.keys())

GREEN_DEPARTURE_PROB = 0.5  # ~30 vehicles/min, one-lane saturation flow
RED_DEPARTURE_PROB = 0.05  # small leak for permitted turns
ARRIVAL_EMA_ALPHA = 0.3  # smoothing for the reported arrival_rate
LANE_CAPACITY = 20  # same normalization the CV adapter and scoring.py use for density/queue


class Simulator:
    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        profile: str = "balanced",
        directions: Tuple[str, ...] = DIRECTIONS,
        rng: Optional[random.Random] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bus = bus
        self.directions = directions
        self._rng = rng or random.Random()
        self._clock = clock
        self.set_profile(profile)
        self._queue_length = {d: self._rng.randint(0, 5) for d in directions}
        self._arrival_ema = {d: 0.0 for d in directions}
        self._green_directions: FrozenSet[str] = frozenset()

    def set_profile(self, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError(f"unknown traffic profile: {profile!r} (choose from {PROFILE_NAMES})")
        self.profile = profile

    def set_green_directions(self, green_directions: FrozenSet[str]) -> None:
        """Used by the live wiring so departures speed up on whichever
        direction currently has a green light."""
        self._green_directions = green_directions

    def tick(self, green_directions: Optional[FrozenSet[str]] = None) -> Dict[str, TrafficMetrics]:
        green = green_directions if green_directions is not None else self._green_directions
        arrival_probs = PROFILES[self.profile]
        batch: Dict[str, TrafficMetrics] = {}
        now = time.time()
        for d in self.directions:
            p = arrival_probs[d]
            if p < 0:
                p = self._rng.uniform(0.1, 0.9)
            arrived = 1 if self._rng.random() < p else 0
            departure_prob = GREEN_DEPARTURE_PROB if d in green else RED_DEPARTURE_PROB
            departed = 1 if self._rng.random() < departure_prob else 0

            self._queue_length[d] = max(0, self._queue_length[d] + arrived - departed)
            queue_length = self._queue_length[d]
            vehicle_count = queue_length + (1 if self._rng.random() < 0.3 else 0)

            instantaneous_rate = arrived * 60.0  # vehicles/min implied by this one tick
            self._arrival_ema[d] = (1 - ARRIVAL_EMA_ALPHA) * self._arrival_ema[d] + ARRIVAL_EMA_ALPHA * instantaneous_rate

            batch[d] = TrafficMetrics(
                direction=d,
                vehicle_count=vehicle_count,
                queue_length=queue_length,
                avg_wait_time=queue_length * 3.0,  # demo estimate: ~3s per queued vehicle
                arrival_rate=self._arrival_ema[d],
                timestamp=now,
                density=min(1.0, vehicle_count / LANE_CAPACITY),
                source_id=f"simulator:{self.profile}",
            )
        return batch

    async def run(self, interval: float = 1.0) -> None:
        if self.bus is None:
            raise RuntimeError("Simulator.run() needs a MessageBus; construct with bus=MessageBus() for live mode")
        while True:
            try:
                await self.bus.publish("metrics", sender="simulator", payload=self.tick())
            except Exception:
                logging.getLogger("maats.cv").exception("Simulator.run(): unexpected error this cycle, continuing")
            await asyncio.sleep(interval)

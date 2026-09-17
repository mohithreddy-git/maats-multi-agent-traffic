"""Deterministic, synchronous fixed-time vs adaptive comparison harness.

The live system drives DirectionalAgent/CoordinatorAgent over the asyncio
MessageBus. For scenario comparisons we want instant, deterministic runs
(no real sleeping, byte-identical traffic between the two arms), so this
harness calls the same agent/coordinator methods directly -- they were
already written as plain synchronous methods in Milestone 1 for exactly
this kind of testability. No agent code changes; this only bypasses the
bus.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from ..agents.coordinator_agent import CoordinatorAgent, DIRECTIONS
from ..agents.directional_agent import DirectionalAgent
from ..agents.message_bus import MessageBus
from ..traffic_engine.fixed_time_controller import FixedTimeController
from ..traffic_engine.signal_fsm import Phase, SignalCommand
from .simulator import Simulator


def _green_directions_for(command: SignalCommand) -> FrozenSet[str]:
    if command.phase != Phase.GREEN:
        return frozenset()
    return frozenset({command.active_direction})


class SyncAdaptiveController:
    """Wires 4 DirectionalAgents + a CoordinatorAgent and drives them by
    direct method calls instead of asyncio message passing."""

    def __init__(self, clock, green_duration: float) -> None:
        bus = MessageBus()  # only needed for construction (listen() subscribes); never published to here
        self.agents: Dict[str, DirectionalAgent] = {
            d: DirectionalAgent(d, bus, clock=clock, peak_override=False) for d in DIRECTIONS
        }
        self.coordinator = CoordinatorAgent(bus, clock=clock, green_duration=green_duration)

    def step(self, batch, now: float) -> SignalCommand:
        for direction, agent in self.agents.items():
            update = agent.process_metrics(batch, now=now)
            self.coordinator.ingest(update)

        transitioned = self.coordinator.fsm.advance(now)
        command = self.coordinator.fsm.command(now)
        if transitioned and command.phase == Phase.GREEN:
            for agent in self.agents.values():
                agent._on_signal_command(command, now=now)
        return command

    def green_directions(self) -> FrozenSet[str]:
        return _green_directions_for(self.coordinator.fsm.command())


@dataclass
class ComparisonSnapshot:
    now: float
    adaptive_command: SignalCommand
    fixed_command: SignalCommand
    adaptive_queues: Dict[str, int]
    fixed_queues: Dict[str, int]
    adaptive_cumulative_vehicle_seconds: float
    fixed_cumulative_vehicle_seconds: float


class ComparisonRunner:
    """Runs the adaptive pipeline and the fixed-time controller side by
    side against identically-seeded traffic (same profile, same seed), so
    any difference in outcomes is attributable to the control strategy, not
    to different traffic."""

    def __init__(
        self,
        profile: str = "balanced",
        seed: int = 42,
        green_duration: float = 20.0,  # same fixed slot length for BOTH arms -- only selection order differs
        dt: float = 1.0,
    ) -> None:
        self.dt = dt
        self._now = 0.0
        clock = lambda: self._now

        self._adaptive_sim = Simulator(profile=profile, rng=random.Random(seed), clock=clock)
        self._fixed_sim = Simulator(profile=profile, rng=random.Random(seed), clock=clock)

        self._adaptive = SyncAdaptiveController(clock=clock, green_duration=green_duration)
        self._fixed = FixedTimeController(green_duration=green_duration, clock=clock)

        self._adaptive_green = self._adaptive.green_directions()
        self._fixed_green = _green_directions_for(self._fixed.fsm.command())

        self.adaptive_cumulative_vehicle_seconds = 0.0
        self.fixed_cumulative_vehicle_seconds = 0.0

    def step(self) -> ComparisonSnapshot:
        self._now += self.dt

        adaptive_batch = self._adaptive_sim.tick(self._adaptive_green)
        fixed_batch = self._fixed_sim.tick(self._fixed_green)

        adaptive_command = self._adaptive.step(adaptive_batch, self._now)
        self._adaptive_green = self._adaptive.green_directions()

        self._fixed.tick(self._now)
        fixed_command = self._fixed.fsm.command(self._now)
        self._fixed_green = _green_directions_for(fixed_command)

        self.adaptive_cumulative_vehicle_seconds += sum(m.queue_length for m in adaptive_batch.values()) * self.dt
        self.fixed_cumulative_vehicle_seconds += sum(m.queue_length for m in fixed_batch.values()) * self.dt

        return ComparisonSnapshot(
            now=self._now,
            adaptive_command=adaptive_command,
            fixed_command=fixed_command,
            adaptive_queues={d: adaptive_batch[d].queue_length for d in DIRECTIONS},
            fixed_queues={d: fixed_batch[d].queue_length for d in DIRECTIONS},
            adaptive_cumulative_vehicle_seconds=self.adaptive_cumulative_vehicle_seconds,
            fixed_cumulative_vehicle_seconds=self.fixed_cumulative_vehicle_seconds,
        )

    def run(self, ticks: int) -> List[ComparisonSnapshot]:
        return [self.step() for _ in range(ticks)]

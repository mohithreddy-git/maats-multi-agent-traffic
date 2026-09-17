"""Coordinator: collects one PriorityUpdate per direction and decides which
single direction gets the NEXT green slot, AND for how long (respecting
starvation), bounded by [MIN_GREEN_SECONDS, GREEN_DURATION_SECONDS].

This is the only place traffic data is allowed to influence the signal at
all. It is still narrow in the way that matters for safety: the coordinator
never touches more than one direction's state at a time, and the duration
it computes is only ever a REQUEST -- the SignalFSM alone owns phase
timing/transitions and clamps/applies it (see AGENTS -> COORDINATOR ->
NEXT DIRECTION + DURATION -> SIGNAL CONTROLLER separation in
signal_fsm.py's docstring). The coordinator cannot interrupt an
in-progress green -- see SignalFSM.request_next()."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import logging

from .base_agent import BaseAgent
from .message_bus import AgentMessage, MessageBus
from ..traffic_engine.metrics import PriorityUpdate
from ..traffic_engine.scoring import compute_green_time
from ..traffic_engine.signal_fsm import GREEN_DURATION_SECONDS, SignalCommand, SignalFSM

DIRECTIONS: Tuple[str, ...] = ("N", "S", "E", "W")

logger = logging.getLogger("maats.signal")


@dataclass(frozen=True)
class SignalDecision:
    """The coordinator's explicit, inspectable decision -- built fresh from
    real PriorityUpdate data every time all directions have reported in.
    Never fabricated: current_direction is always the FSM's real active
    direction, next_direction/green_duration are always decide()'s real
    output, and traffic_summary is always this round's real vehicle counts."""

    current_direction: str
    next_direction: str
    green_duration: float  # seconds, calculated from next_direction's own real traffic metrics
    traffic_summary: Dict[str, int]  # direction -> vehicle_count, this round
    reason: str  # "starvation_override" or "highest_priority"
    timestamp: float


class CoordinatorAgent(BaseAgent):
    def __init__(
        self,
        bus: MessageBus,
        directions: Tuple[str, ...] = DIRECTIONS,
        clock: Callable[[], float] = time.monotonic,
        initial_direction: Optional[str] = None,
        green_duration: float = GREEN_DURATION_SECONDS,
    ) -> None:
        super().__init__(name="coordinator", bus=bus)
        self.directions = directions
        self._clock = clock
        self._latest: Dict[str, PriorityUpdate] = {}
        self.fsm = SignalFSM(
            directions=directions, initial_direction=initial_direction, green_duration=green_duration, clock=clock,
        )
        self.last_decision: Optional[SignalDecision] = None
        self.listen("priority_updates")

    def ingest(self, update: PriorityUpdate) -> None:
        self._latest[update.direction] = update
        if len(self._latest) == len(self.directions):
            direction, green_duration, starvation = self.decide()
            self.fsm.request_next(direction, green_duration=green_duration, starvation=starvation)
            self.last_decision = SignalDecision(
                current_direction=self.fsm.active_direction,
                next_direction=direction,
                green_duration=green_duration,
                traffic_summary={d: u.vehicle_count for d, u in self._latest.items()},
                reason="starvation_override" if starvation else "highest_priority",
                timestamp=self._clock(),
            )

    def decide(self) -> Tuple[str, float, bool]:
        """Picks which direction gets the NEXT slot, and for how long. A
        starved direction always wins the DIRECTION outright, regardless of
        score; otherwise the highest-priority direction wins. Either way,
        the DURATION is always calculated from the winning direction's own
        real traffic metrics (never fabricated, never cross-direction) via
        compute_green_time(), bounded to [MIN_GREEN, MAX_GREEN] -- starvation
        affects who goes next, never how long they get."""
        updates = self._latest
        starved = [u for u in updates.values() if u.is_starved]
        if starved:
            winner = max(starved, key=lambda u: u.time_since_last_green)
            starvation = True
        else:
            winner = max(updates.values(), key=lambda u: u.priority_score)
            starvation = False
        green_duration = compute_green_time(
            vehicle_count=winner.vehicle_count, queue_length=winner.queue_length, density=winner.density,
        )
        return winner.direction, green_duration, starvation

    async def handle_message(self, message: AgentMessage) -> None:
        self.ingest(message.payload)

    async def tick(self, now: Optional[float] = None) -> Optional[SignalCommand]:
        """Advance the FSM's clock; returns a SignalCommand if a transition
        just happened, so the caller can publish it."""
        now = now if now is not None else self._clock()
        if self.fsm.advance(now):
            command = self.fsm.command(now)
            # Runtime invariant logging: the FSM's own data model (a single
            # active_direction field) makes two simultaneous greens
            # unrepresentable, but this is the one place every real
            # transition is observable in production -- log it so a
            # violation of "exactly one green" would show up immediately in
            # ops logs, not just in tests.
            green_count = sum(1 for d in self.directions if self.fsm.light_for(d, now) == "GREEN")
            if green_count > 1:
                logger.error(
                    "SIGNAL INVARIANT VIOLATION: %d directions GREEN simultaneously at t=%.1f (phase=%s, active=%s)",
                    green_count, now, command.phase.value, command.active_direction,
                )
            else:
                logger.info(
                    "phase=%s active=%s duration=%.0fs starvation=%s",
                    command.phase.value, command.active_direction, command.duration, command.starvation_override,
                )
            return command
        return None

    async def run(self) -> None:
        self._running = True

        async def tick_loop(interval: float = 0.5) -> None:
            while self._running:
                await asyncio.sleep(interval)
                command = await self.tick()
                if command is not None:
                    await self.publish("signal_command", command)

        await asyncio.gather(super().run(), tick_loop())

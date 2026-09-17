"""Independent per-direction agent: owns its own state, updates it only from
messages it receives, and never touches another agent's attributes."""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from .base_agent import BaseAgent
from .message_bus import AgentMessage, MessageBus
from ..traffic_engine.metrics import PriorityUpdate, TrafficMetrics
from ..traffic_engine.scoring import compute_priority_score, is_peak_hour_now, is_starved
from ..traffic_engine.signal_fsm import Phase, SignalCommand


class DirectionalAgent(BaseAgent):
    def __init__(
        self,
        direction: str,
        bus: MessageBus,
        clock: Callable[[], float] = time.monotonic,
        peak_override: Optional[bool] = None,
    ) -> None:
        super().__init__(name=f"agent.{direction}", bus=bus)
        self.direction = direction
        self.vehicle_count = 0
        self.queue_length = 0
        self.avg_wait_time = 0.0
        self.arrival_rate = 0.0
        self.density = 0.0
        self.priority_score = 0.0
        self._clock = clock
        self._last_green_at = clock()
        self._peak_override = peak_override
        self.listen("metrics", "signal_command")

    @property
    def time_since_last_green(self) -> float:
        return self._clock() - self._last_green_at

    def mark_green(self, now: Optional[float] = None) -> None:
        self._last_green_at = now if now is not None else self._clock()

    def process_metrics(self, batch: Dict[str, TrafficMetrics], now: Optional[float] = None) -> PriorityUpdate:
        now = now if now is not None else self._clock()
        mine = batch[self.direction]
        self.vehicle_count = mine.vehicle_count
        self.queue_length = mine.queue_length
        self.avg_wait_time = mine.avg_wait_time
        self.arrival_rate = mine.arrival_rate
        self.density = mine.density

        total_vehicle_count = sum(m.vehicle_count for m in batch.values())
        time_since_last_green = now - self._last_green_at
        is_peak = self._peak_override if self._peak_override is not None else is_peak_hour_now()

        self.priority_score = compute_priority_score(
            queue_length=self.queue_length,
            time_since_last_green=time_since_last_green,
            arrival_rate=self.arrival_rate,
            vehicle_count=self.vehicle_count,
            total_vehicle_count=total_vehicle_count,
            is_peak=is_peak,
        )
        return PriorityUpdate(
            direction=self.direction,
            priority_score=self.priority_score,
            vehicle_count=self.vehicle_count,
            queue_length=self.queue_length,
            avg_wait_time=self.avg_wait_time,
            arrival_rate=self.arrival_rate,
            time_since_last_green=time_since_last_green,
            is_starved=is_starved(time_since_last_green),
            timestamp=time.time(),
            density=self.density,
        )

    def _on_signal_command(self, command: SignalCommand, now: Optional[float] = None) -> None:
        if command.phase == Phase.GREEN and self.direction == command.active_direction:
            self.mark_green(now)

    async def handle_message(self, message: AgentMessage) -> None:
        if message.topic == "metrics":
            update = self.process_metrics(message.payload)
            await self.publish("priority_updates", update)
        elif message.topic == "signal_command":
            self._on_signal_command(message.payload)

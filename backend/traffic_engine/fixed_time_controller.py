"""Fixed-time baseline controller: ignores traffic entirely and rotates
through the configured direction sequence (NORTH -> EAST -> SOUTH -> WEST
by default) on a constant cycle. Reuses SignalFSM so the safety timing
(yellow, all-red, single-direction-only) is identical to the adaptive
controller -- this class adds nothing on top of SignalFSM except the
guarantee that request_next() is never called, which is exactly what makes
it deterministic round-robin rather than priority-driven. This is also
what "no valid video input / no traffic data" degrades to: the coordinator
simply never calls request_next() either, so SignalFSM's own default
rotation takes over -- there is no separate "stalled" state to fall into.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from .signal_fsm import DEFAULT_DIRECTION_SEQUENCE, GREEN_DURATION_SECONDS, SignalCommand, SignalFSM


class FixedTimeController:
    def __init__(
        self,
        directions: Tuple[str, ...] = DEFAULT_DIRECTION_SEQUENCE,
        green_duration: float = GREEN_DURATION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fsm = SignalFSM(directions=directions, green_duration=green_duration, clock=clock)

    def tick(self, now: Optional[float] = None) -> Optional[SignalCommand]:
        if not self.fsm.advance(now):
            return None
        return self.fsm.command(now)

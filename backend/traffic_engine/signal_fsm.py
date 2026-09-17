"""Signal phase state machine -- STRICT SINGLE-DIRECTION CONTROL.

At any instant exactly one direction is GREEN (or YELLOW, mid-transition)
and the other three are RED. This is enforced structurally, not by
convention: the FSM only ever tracks one `active_direction`, so there is no
representable state with two greens -- unlike the old NS/EW paired-axis
model this replaces, which made N+S (or E+W) green *together* by design.
This invariant is completely independent of green duration (below) --
nothing about making the duration adaptive touches how many directions can
ever be non-RED at once.

Sequence: GREEN(direction) -> YELLOW(direction) -> ALL_RED -> GREEN(next).
`next` (and its duration) is only resolved when the FSM actually exits
ALL_RED, so a mid-green `request_next()` call (e.g. because a traffic
score just changed) can never interrupt the in-progress green/yellow phase
-- it only ever affects the *next* phase: which direction runs, and for
how long.

Green duration is ADAPTIVE, bounded by [MIN_GREEN_SECONDS,
GREEN_DURATION_SECONDS]: the coordinator computes it once per slot (see
scoring.compute_green_time(), from that direction's own real traffic
metrics) and passes it to request_next(); the FSM locks it in for the
whole slot once applied -- it never changes mid-phase. When nobody calls
request_next() at all (no video / no traffic data), the FSM falls back to
its own constructor default, which is GREEN_DURATION_SECONDS (90s) for
every direction -- this is the "no-video fallback" behavior.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

GREEN_DURATION_SECONDS = 90.0  # MAXIMUM green duration, and the no-video/no-metrics fallback duration
MIN_GREEN_SECONDS = 10.0  # sensible configurable minimum -- see scoring.compute_green_time()
YELLOW_DURATION = 3.0
ALL_RED_DURATION = 1.0

# NORTH -> EAST -> SOUTH -> WEST -> NORTH -> ... is the default rotation used
# whenever nobody calls request_next() (no video / no traffic data). Order
# matters here (unlike DIRECTIONS elsewhere, which is just "which 4 exist").
DEFAULT_DIRECTION_SEQUENCE: Tuple[str, ...] = ("N", "E", "S", "W")


class Phase(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


@dataclass(frozen=True)
class SignalCommand:
    phase: Phase
    active_direction: str  # the direction currently GREEN/YELLOW; RED for everyone else, always
    started_at: float
    duration: float
    time_remaining: float
    starvation_override: bool


class SignalFSM:
    def __init__(
        self,
        directions: Tuple[str, ...] = DEFAULT_DIRECTION_SEQUENCE,
        initial_direction: Optional[str] = None,
        green_duration: float = GREEN_DURATION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if len(directions) < 2:
            raise ValueError("need at least 2 directions to rotate through")
        self._directions = tuple(directions)
        self._clock = clock
        now = clock()
        self._green_duration = green_duration
        self._active_direction = initial_direction or self._directions[0]
        if self._active_direction not in self._directions:
            raise ValueError(f"initial_direction {self._active_direction!r} not in {self._directions!r}")
        self._phase = Phase.GREEN
        self._phase_started_at = now
        self._duration = green_duration
        self._starvation = False
        self._pending_direction: Optional[str] = None
        self._pending_green_duration: Optional[float] = None
        self._pending_starvation = False

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def active_direction(self) -> str:
        return self._active_direction

    @property
    def directions(self) -> Tuple[str, ...]:
        return self._directions

    def time_remaining(self, now: Optional[float] = None) -> float:
        now = now if now is not None else self._clock()
        return max(0.0, self._duration - (now - self._phase_started_at))

    def request_next(self, direction: str, green_duration: Optional[float] = None, starvation: bool = False) -> None:
        """Records which direction (and, optionally, for how long) should
        get the NEXT green slot. Only applied the next time the FSM
        actually exits ALL_RED (see advance()), so calling this
        mid-green/yellow -- e.g. because a coordinator just recomputed
        priorities from new traffic data -- can never interrupt the phase
        in progress; it only ever affects the phase that starts *after*
        this one finishes.

        green_duration=None means "use this FSM's own default" (the
        no-video/no-traffic-data fallback, GREEN_DURATION_SECONDS). When
        given, it is clamped to [MIN_GREEN_SECONDS, GREEN_DURATION_SECONDS]
        here -- the FSM is the final safety authority on timing bounds, so
        a caller can never push a slot below the minimum or above the
        maximum even by mistake."""
        if direction not in self._directions:
            raise ValueError(f"unknown direction: {direction!r}; choose from {self._directions!r}")
        self._pending_direction = direction
        if green_duration is not None:
            green_duration = max(MIN_GREEN_SECONDS, min(GREEN_DURATION_SECONDS, green_duration))
        self._pending_green_duration = green_duration
        self._pending_starvation = starvation

    def _default_next_direction(self) -> str:
        # No video / no traffic data / nobody called request_next this cycle
        # -> strict round-robin through the configured sequence. This is
        # what makes "never stall" true even with zero CV input.
        idx = self._directions.index(self._active_direction)
        return self._directions[(idx + 1) % len(self._directions)]

    def advance(self, now: Optional[float] = None) -> bool:
        """Moves to the next phase if the current phase's duration has
        elapsed. Returns True if a transition happened. Never skips more
        than one phase per call, even if `now` jumps far ahead -- so a
        caller polling irregularly still sees every GREEN->YELLOW->ALL_RED
        step, never a direct GREEN->GREEN jump."""
        now = now if now is not None else self._clock()
        if now - self._phase_started_at < self._duration:
            return False

        if self._phase == Phase.GREEN:
            self._phase = Phase.YELLOW
            self._duration = YELLOW_DURATION
        elif self._phase == Phase.YELLOW:
            self._phase = Phase.ALL_RED
            self._duration = ALL_RED_DURATION
        else:  # ALL_RED -> GREEN(next) -- the only place active_direction changes
            next_direction = self._pending_direction or self._default_next_direction()
            if self._pending_direction:
                self._starvation = self._pending_starvation
                next_duration = self._pending_green_duration if self._pending_green_duration is not None else self._green_duration
            else:
                # nobody called request_next this cycle (no video/no traffic
                # data) -- deterministic fallback: this FSM's own configured
                # default duration, never a calculated one.
                self._starvation = False
                next_duration = self._green_duration
            self._active_direction = next_direction
            self._phase = Phase.GREEN
            self._duration = next_duration
            self._pending_direction = None
            self._pending_green_duration = None
            self._pending_starvation = False

        self._phase_started_at = now
        return True

    def command(self, now: Optional[float] = None) -> SignalCommand:
        now = now if now is not None else self._clock()
        return SignalCommand(
            phase=self._phase,
            active_direction=self._active_direction,
            started_at=self._phase_started_at,
            duration=self._duration,
            time_remaining=self.time_remaining(now),
            starvation_override=self._starvation,
        )

    def light_for(self, direction: str, now: Optional[float] = None) -> str:
        """RED/YELLOW/GREEN for one direction. Exactly one direction can
        ever be non-RED -- there is only one active_direction, so two
        simultaneous greens are structurally impossible, not just avoided
        by convention."""
        if direction != self._active_direction:
            return "RED"
        if self._phase == Phase.GREEN:
            return "GREEN"
        if self._phase == Phase.YELLOW:
            return "YELLOW"
        return "RED"  # ALL_RED

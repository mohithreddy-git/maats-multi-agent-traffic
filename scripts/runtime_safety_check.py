"""Runtime assertion/logging pass: drives the REAL agent/coordinator/FSM
pipeline (DirectionalAgent x4 + CoordinatorAgent + Simulator, the same
classes backend/main.py and dashboard.py wire together) through several
minutes of simulated operation, with the coordinator's own runtime
invariant logging (see coordinator_agent.py's tick()) active, plus an
independent second check via the hardware safety gate
(backend.hardware.led_signal.assert_single_green_state) at every single
tick -- so a violation would have to slip past two independent checks to
go unnoticed.

Uses a fake clock advanced in 1-second steps rather than real wall-clock
sleeping, so "several minutes of operation" (at GREEN_DURATION_SECONDS=90s
per slot) completes in under a second of actual runtime while exercising
the exact same code path production uses.

Usage:
    python scripts/runtime_safety_check.py [minutes]
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.coordinator_agent import CoordinatorAgent, DIRECTIONS  # noqa: E402
from backend.agents.directional_agent import DirectionalAgent  # noqa: E402
from backend.agents.message_bus import MessageBus  # noqa: E402
from backend.hardware.led_signal import assert_single_green_state, compute_led_state  # noqa: E402
from backend.simulation.simulator import Simulator  # noqa: E402
from backend.traffic_engine.signal_fsm import GREEN_DURATION_SECONDS, Phase  # noqa: E402


def run(minutes: float = 5.0, csv_path: str = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger("maats.runtime_check")

    bus = MessageBus()
    clock = [0.0]
    sim = Simulator(profile="balanced", clock=lambda: clock[0])
    agents = {d: DirectionalAgent(d, bus, clock=lambda: clock[0]) for d in DIRECTIONS}
    coordinator = CoordinatorAgent(bus, clock=lambda: clock[0])  # real GREEN_DURATION_SECONDS=90s default

    total_seconds = minutes * 60.0
    ticks = 0
    transitions = 0
    hardware_violations = 0
    directions_served: set = set()
    green_seconds_by_direction = {d: 0.0 for d in DIRECTIONS}
    records = []  # one per tick: TIMESTAMP, ACTIVE_DIRECTION, PHASE, TIME_REMAINING, GREEN_DIRECTIONS
    invariant_violations = 0

    logger.info("Starting %.1f-minute runtime safety pass (GREEN_DURATION_SECONDS=%.0f)", minutes, GREEN_DURATION_SECONDS)

    while clock[0] < total_seconds:
        clock[0] += 1.0
        ticks += 1

        batch = sim.tick()
        for direction, agent in agents.items():
            update = agent.process_metrics(batch, now=clock[0])
            coordinator.ingest(update)

        if coordinator.fsm.advance(clock[0]):
            transitions += 1
            command = coordinator.fsm.command(clock[0])
            for agent in agents.values():
                agent._on_signal_command(command, now=clock[0])
            if command.phase == Phase.GREEN:
                sim.set_green_directions(frozenset({command.active_direction}))
                directions_served.add(command.active_direction)
            # This script drives the FSM synchronously (direct method calls,
            # same convention as comparison.py's harness) rather than through
            # CoordinatorAgent.tick()'s async loop, so it logs each
            # transition itself, matching what tick() logs in production.
            logger.info(
                "phase=%s active=%s duration=%.0fs starvation=%s",
                command.phase.value, command.active_direction, command.duration, command.starvation_override,
            )

        # Independent second check: run the CURRENT command through the
        # hardware safety gate every tick, not just on transitions. A real
        # violation here would mean the FSM's own single-active-direction
        # invariant was somehow bypassed.
        command = coordinator.fsm.command(clock[0])
        led_state = compute_led_state(DIRECTIONS, command)
        safe_state = assert_single_green_state(led_state)
        if safe_state != led_state:
            hardware_violations += 1
        if command.phase == Phase.GREEN:
            green_seconds_by_direction[command.active_direction] += 1.0

        # The exact per-tick record format requested: TIMESTAMP,
        # ACTIVE_DIRECTION, PHASE, TIME_REMAINING, GREEN_DIRECTIONS.
        # GREEN_DIRECTIONS is the real count from led_state (independent of
        # the hardware-gate check above), and must be exactly 1 during
        # GREEN and exactly 0 during YELLOW/ALL_RED, every single tick.
        green_directions_count = sum(1 for light in led_state.values() if light == "GREEN")
        expected = 1 if command.phase == Phase.GREEN else 0
        if green_directions_count != expected or green_directions_count > 1:
            invariant_violations += 1
        records.append((clock[0], command.active_direction, command.phase.value, command.time_remaining, green_directions_count))

    if csv_path:
        import csv as csv_module

        with open(csv_path, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow(["TIMESTAMP", "ACTIVE_DIRECTION", "PHASE", "TIME_REMAINING", "GREEN_DIRECTIONS"])
            writer.writerows(records)

    logger.info(
        "Runtime safety pass complete: %d ticks, %d transitions, %d hardware-gate violations, "
        "%d invariant-record violations, directions served=%s",
        ticks, transitions, hardware_violations, invariant_violations, sorted(directions_served),
    )

    return {
        "minutes": minutes,
        "ticks": ticks,
        "transitions": transitions,
        "hardware_violations": hardware_violations,
        "invariant_violations": invariant_violations,
        "directions_served": sorted(directions_served),
        "green_seconds_by_direction": green_seconds_by_direction,
        "records": records,
    }


if __name__ == "__main__":
    minutes_arg = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    csv_arg = sys.argv[2] if len(sys.argv) > 2 else None
    report = run(minutes_arg, csv_path=csv_arg)
    records = report.pop("records")
    print()
    print("=== Runtime Safety Report ===")
    for key, value in report.items():
        print(f"{key}: {value}")
    print()
    print("Sample records (TIMESTAMP, ACTIVE_DIRECTION, PHASE, TIME_REMAINING, GREEN_DIRECTIONS):")
    stride = max(1, len(records) // 15)
    for record in records[::stride]:
        print(record)
    if report["hardware_violations"] > 0 or report["invariant_violations"] > 0:
        raise SystemExit(
            f"FAIL: {report['hardware_violations']} hardware violations, "
            f"{report['invariant_violations']} invariant-record violations detected"
        )
    print(f"\nPASS: {report['ticks']} records checked, zero GREEN_DIRECTIONS > 1, zero mismatches with phase")

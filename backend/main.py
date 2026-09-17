"""Console-only end-to-end wiring: traffic source -> 4 DirectionalAgents ->
CoordinatorAgent -> printed SignalCommands. No dashboard yet.

The traffic source is interchangeable: agents only ever see TrafficMetrics,
never who produced them.

Usage:
  python -m backend.main sim [profile]          (default: sim balanced)
  python -m backend.main cv [roi_config.json]    (default: the single-direction demo config)
"""
from __future__ import annotations

import asyncio
import sys

from .agents.coordinator_agent import CoordinatorAgent, DIRECTIONS
from .agents.directional_agent import DirectionalAgent
from .agents.message_bus import MessageBus
from .simulation.simulator import PROFILE_NAMES, Simulator
from .traffic_engine.signal_fsm import Phase

DEFAULT_CV_CONFIG = "backend/cv/configs/demo_single_direction.json"


def _build_source(bus: MessageBus, args: list[str]):
    source_kind = args[0] if args else "sim"

    if source_kind == "cv":
        from .cv.metrics_adapter import VisionSource
        from .cv.roi_config import load_roi_configs

        config_path = args[1] if len(args) > 1 else DEFAULT_CV_CONFIG
        configs = load_roi_configs(config_path)
        return VisionSource(bus, configs)

    if source_kind == "sim":
        profile = args[1] if len(args) > 1 else "balanced"
        if profile not in PROFILE_NAMES:
            raise SystemExit(f"unknown profile {profile!r}; choose from {PROFILE_NAMES}")
        return Simulator(bus, profile=profile)

    raise SystemExit(f"unknown source {source_kind!r}; use 'sim' or 'cv'")


async def main() -> None:
    bus = MessageBus()
    source = _build_source(bus, sys.argv[1:])
    # Derive the active directions from the source itself (a single-camera
    # CV demo may only cover "N") rather than assuming all four -- an agent
    # with no metrics for its direction would otherwise crash on KeyError.
    directions = getattr(source, "directions", DIRECTIONS)
    directional_agents = [DirectionalAgent(d, bus) for d in directions]
    coordinator = CoordinatorAgent(bus, directions=directions)

    async def track_green_directions() -> None:
        """Feeds the current green axis back into the source so queues
        actually drain when their light is green (the simulator uses this
        for realistic dynamics; the CV adapter ignores it since real
        traffic drains on its own) -- pure message passing, no shared
        object between the coordinator and the source."""
        if not hasattr(source, "set_green_directions"):
            return
        queue = bus.subscribe("signal_command")
        while True:
            message = await queue.get()
            command = message.payload
            if command.phase == Phase.GREEN:
                source.set_green_directions(frozenset({command.active_direction}))

    async def print_signal_commands() -> None:
        queue = bus.subscribe("signal_command")
        while True:
            message = await queue.get()
            cmd = message.payload
            print(
                f"[{cmd.phase.value}] active={cmd.active_direction} "
                f"duration={cmd.duration:.0f}s remaining={cmd.time_remaining:.0f}s "
                f"starvation={cmd.starvation_override}"
            )

    await asyncio.gather(
        *(agent.run() for agent in directional_agents),
        coordinator.run(),
        source.run(),
        track_green_directions(),
        print_signal_commands(),
    )


if __name__ == "__main__":
    asyncio.run(main())

"""Behavioral proof of the full chain: increased East traffic -> East's
DirectionalAgent raises its priority -> a real PriorityUpdate message is
published on the MessageBus -> CoordinatorAgent.handle_message consumes it
-> the FSM is eventually driven to East becoming the active direction ->
the SignalCommand message is published and each DirectionalAgent's own
handle_message resets its wait clock.

Every step below calls the *real* handle_message()/publish() methods used in
production (backend/main.py, dashboard.py) -- nothing is bypassed into a
plain dict or a fake message. The only thing this test controls that
production doesn't is time itself (an injected fake clock, ticked by hand),
which is what makes it fast and deterministic instead of needing real
seconds of sleep to watch a signal cycle complete.
"""
from __future__ import annotations

import asyncio
import random

from backend.agents.coordinator_agent import CoordinatorAgent, DIRECTIONS
from backend.agents.directional_agent import DirectionalAgent
from backend.agents.message_bus import MessageBus
from backend.simulation.simulator import Simulator
from backend.traffic_engine.signal_fsm import Phase

INJECT_AT_STEP = 150
# With adaptive durations, a single slot can now run up to MAX_GREEN (90s)
# instead of the old fixed 10s -- fewer rotations happen per unit simulated
# time, so this window must be generous enough for East to actually get a
# turn after its traffic ramps up at INJECT_AT_STEP.
TOTAL_STEPS = 2000


def test_increasing_east_traffic_flips_the_coordinator_to_east_and_changes_the_signal():
    trace: list[str] = []

    async def scenario():
        clock_time = [0.0]
        clock = lambda: clock_time[0]

        bus = MessageBus()
        sim = Simulator(profile="balanced", rng=random.Random(11), clock=clock)
        agents = {d: DirectionalAgent(d, bus, clock=clock) for d in DIRECTIONS}
        coordinator = CoordinatorAgent(bus, clock=clock, initial_direction="N", green_duration=10.0)

        east_priority_history: list[float] = []
        commands = []

        for step in range(TOTAL_STEPS):
            clock_time[0] += 1.0

            if step == INJECT_AT_STEP:
                sim.set_profile("heavy_east")
                trace.append(f"t={clock_time[0]:.0f}s  TEST INJECTS: East traffic increased (profile -> heavy_east)")

            # STAGE 1: VIDEO/SIM -> TrafficMetrics, published as a real bus message
            batch = sim.tick()
            await bus.publish("metrics", sender="simulator", payload=batch)
            trace.append(
                f"t={clock_time[0]:.0f}s  SIM → BUS[metrics]: East(count={batch['E'].vehicle_count}, "
                f"queue={batch['E'].queue_length})"
            )

            # STAGE 2+3: each DirectionalAgent consumes its own real queued
            # message and updates its own independent state
            for direction, agent in agents.items():
                msg = await agent._queues["metrics"].get()
                await agent.handle_message(msg)
                if direction == "E":
                    east_priority_history.append(agent.priority_score)

            # STAGE 4: each agent's real PriorityUpdate publish, consumed by
            # the Coordinator's real handle_message (== real ingest())
            for _ in range(len(DIRECTIONS)):
                msg = await coordinator._queues["priority_updates"].get()
                if msg.sender == "agent.E":
                    trace.append(
                        f"t={clock_time[0]:.0f}s  EastAgent → Coordinator: "
                        f"{{type: priority_update, direction: 'E', vehicle_count: {msg.payload.vehicle_count}, "
                        f"queue_length: {msg.payload.queue_length}, priority: {msg.payload.priority_score:.2f}}}"
                    )
                await coordinator.handle_message(msg)

            # STAGE 5: FSM decision -> real SignalCommand publish, consumed
            # by every agent's real handle_message
            if coordinator.fsm.advance(clock_time[0]):
                command = coordinator.fsm.command(clock_time[0])
                await bus.publish("signal_command", sender="coordinator", payload=command)
                for agent in agents.values():
                    msg = await agent._queues["signal_command"].get()
                    await agent.handle_message(msg)
                commands.append(command)
                if command.phase == Phase.GREEN:
                    # same feedback wiring dashboard.py/main.py use so a
                    # direction's queue actually drains once it gets green
                    sim.set_green_directions(frozenset({command.active_direction}))
                trace.append(
                    f"t={clock_time[0]:.0f}s  Coordinator → BUS[signal_command]: phase={command.phase.value} "
                    f"active={command.active_direction} duration={command.duration}"
                )

        return east_priority_history, commands

    east_priority_history, commands = asyncio.run(scenario())

    print("\n".join(trace))

    pre_injection_max = max(east_priority_history[:INJECT_AT_STEP])
    post_injection_max = max(east_priority_history[INJECT_AT_STEP:])
    assert post_injection_max > pre_injection_max  # East's own priority genuinely rose

    green_commands_after_injection = [
        c for c in commands if c.started_at >= INJECT_AT_STEP and c.phase == Phase.GREEN
    ]
    assert any(c.active_direction == "E" for c in green_commands_after_injection)

    # Every GREEN command's duration must be a real, bounded, calculated
    # value -- never a fabricated or out-of-range number -- regardless of
    # which direction is currently winning.
    from backend.traffic_engine.scoring import MAX_GREEN, MIN_GREEN

    for command in green_commands_after_injection:
        assert MIN_GREEN <= command.duration <= MAX_GREEN

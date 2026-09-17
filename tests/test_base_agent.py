"""Regression test for the adversarial-audit fix: one malformed message
must not kill an agent's run() task. Every agent runs inside the same
asyncio.gather() as the coordinator and the traffic source (see
dashboard.py/main.py) -- an unhandled exception in handle_message used to
propagate out of run(), which asyncio.gather() (default, no
return_exceptions=True) turns into cancelling every sibling task,
including the coordinator that drives the signal. That would freeze the
signal mid-slot with no way back to normal operation.
"""
from __future__ import annotations

import asyncio

from backend.agents.base_agent import BaseAgent
from backend.agents.message_bus import MessageBus


class _FlakyAgent(BaseAgent):
    """Raises on its first message, then works normally -- simulates a
    single malformed metrics batch or similar transient bad input."""

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(name="flaky", bus=bus)
        self.processed: list = []
        self._calls = 0
        self.listen("test_topic")

    async def handle_message(self, message) -> None:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("simulated malformed message")
        self.processed.append(message.payload)


def test_one_bad_message_does_not_kill_the_agents_run_task():
    async def scenario():
        bus = MessageBus()
        agent = _FlakyAgent(bus)
        task = asyncio.create_task(agent.run())

        await bus.publish("test_topic", sender="test", payload="bad")  # triggers the raise internally
        await bus.publish("test_topic", sender="test", payload="good-1")
        await bus.publish("test_topic", sender="test", payload="good-2")

        for _ in range(50):
            await asyncio.sleep(0)
            if len(agent.processed) == 2:
                break

        assert agent.processed == ["good-1", "good-2"]  # kept working after the bad message
        assert not task.done()  # run() is still alive, not crashed

        agent.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

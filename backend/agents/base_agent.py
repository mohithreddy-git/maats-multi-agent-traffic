"""Common asyncio message-loop plumbing shared by all agents.

Subclasses hold their own independent state (plain instance attributes) and
only exchange information through the MessageBus -- never through direct
references to each other's objects.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from .message_bus import AgentMessage, MessageBus

logger = logging.getLogger("maats.agents")


class BaseAgent(ABC):
    def __init__(self, name: str, bus: MessageBus) -> None:
        self.name = name
        self.bus = bus
        self._queues: dict[str, asyncio.Queue] = {}
        self._running = False

    def listen(self, *topics: str) -> None:
        for topic in topics:
            self._queues[topic] = self.bus.subscribe(topic)

    async def publish(self, topic: str, payload) -> None:
        await self.bus.publish(topic, sender=self.name, payload=payload)

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> None:
        ...

    async def run(self) -> None:
        if not self._queues:
            raise RuntimeError(f"{self.name} has no subscriptions; call listen() first")
        self._running = True
        pending = {topic: asyncio.create_task(queue.get()) for topic, queue in self._queues.items()}
        while self._running:
            done, _ = await asyncio.wait(pending.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                topic = next(t for t, tsk in pending.items() if tsk is task)
                try:
                    await self.handle_message(task.result())
                except Exception:
                    # One malformed message must not kill this agent's task --
                    # every agent runs inside the same asyncio.gather() as the
                    # coordinator and the traffic source (see dashboard.py/
                    # main.py), so an unhandled exception here would otherwise
                    # cancel the ENTIRE pipeline, freezing the signal mid-slot
                    # with no path back to normal operation.
                    logger.exception("%s: unhandled error processing a %r message, continuing", self.name, topic)
                pending[topic] = asyncio.create_task(self._queues[topic].get())

    def stop(self) -> None:
        # ponytail: doesn't cancel the in-flight queue.get() tasks, so run()
        # only actually exits after its next message. Fine for a demo driven
        # by asyncio.run(); revisit with task cancellation if that matters.
        # 
        self._running = False

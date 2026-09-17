"""Asyncio pub/sub bus.

Each subscriber gets its own asyncio.Queue -- fan-out delivery, not a shared
mutable structure that agents read into and mutate. Agents only ever see
their own copy of a message.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class AgentMessage:
    topic: str
    sender: str
    payload: Any
    timestamp: float


class MessageBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List["asyncio.Queue[AgentMessage]"]] = {}

    def subscribe(self, topic: str) -> "asyncio.Queue[AgentMessage]":
        queue: "asyncio.Queue[AgentMessage]" = asyncio.Queue()
        self._subscribers.setdefault(topic, []).append(queue)
        return queue

    async def publish(self, topic: str, sender: str, payload: Any) -> None:
        message = AgentMessage(topic=topic, sender=sender, payload=payload, timestamp=time.time())
        for queue in self._subscribers.get(topic, []):
            await queue.put(message)

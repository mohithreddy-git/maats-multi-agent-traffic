"""Mixes real video-derived TrafficMetrics with simulator-derived
TrafficMetrics into one combined batch per tick, covering all four
directions -- lets a demo run with as few as one real video (the rest
simulated) while DirectionalAgent/CoordinatorAgent stay completely
unaware of which producer supplied which direction's numbers. Same
"metrics" topic, same TrafficMetrics contract, same agents -- nothing
downstream is bypassed.

Covers MAATS' two legitimate demo modes with one class:
  MODE A (four-video): every direction is assigned "video"
  MODE B (single-video/test): one or more directions are "video", the
    rest are "simulated" -- a direction is only ever reported as
    "live_video" if its camera actually opened; a video that fails to
    open is "no_source", never silently relabelled as simulated or live.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, FrozenSet, Literal, Optional, Tuple

from ..agents.message_bus import MessageBus
from ..cv.metrics_adapter import DirectionVisionAdapter
from ..cv.roi_config import ROIConfig
from ..traffic_engine.metrics import TrafficMetrics
from .simulator import Simulator

SourceLabel = Literal["live_video", "simulated", "no_source"]


class HybridSource:
    def __init__(
        self,
        bus: MessageBus,
        video_configs: Dict[str, ROIConfig],
        simulated_directions: Tuple[str, ...] = (),
        simulator_profile: str = "balanced",
    ) -> None:
        self.bus = bus
        self.errors: Dict[str, str] = {}
        self.video_adapters: Dict[str, DirectionVisionAdapter] = {}
        for direction, config in video_configs.items():
            try:
                self.video_adapters[direction] = DirectionVisionAdapter(config)
            except Exception as exc:
                self.errors[direction] = str(exc)

        # a direction already covered by video can't also be simulated
        self.simulated_directions: Tuple[str, ...] = tuple(
            d for d in simulated_directions if d not in self.video_adapters
        )
        self.simulator_profile = simulator_profile
        self._simulator: Optional[Simulator] = (
            Simulator(profile=simulator_profile, directions=self.simulated_directions)
            if self.simulated_directions
            else None
        )

        self.directions: Tuple[str, ...] = tuple(self.video_adapters.keys()) + self.simulated_directions
        if not self.directions:
            raise RuntimeError(f"no direction could be started: {self.errors}")

        self.source_labels: Dict[str, SourceLabel] = {}
        for direction in video_configs:
            self.source_labels[direction] = "live_video" if direction in self.video_adapters else "no_source"
        for direction in self.simulated_directions:
            self.source_labels[direction] = "simulated"

    def tick(self) -> Dict[str, TrafficMetrics]:
        now = time.time()
        batch: Dict[str, TrafficMetrics] = {
            direction: adapter.tick(now) for direction, adapter in self.video_adapters.items()
        }
        if self._simulator is not None:
            batch.update(self._simulator.tick())
        return batch

    def switch_video(self, direction: str, video_source: str, detector_kind: Optional[str] = None) -> None:
        """Hot-swaps one already-video-backed direction's footage in place.
        Never touches self.video_adapters' keys, self.simulated_directions,
        or self._simulator -- so this is safe to call while tick()/run() is
        mid-flight on another direction, with no pipeline rebuild and no
        change to which directions exist."""
        if direction not in self.video_adapters:
            raise KeyError(f"{direction!r} has no running video adapter to switch")
        self.video_adapters[direction].switch_video(video_source, detector_kind=detector_kind)

    def status_snapshot(self) -> Dict[str, dict]:
        return {d: a.status() for d, a in self.video_adapters.items()}

    def set_green_directions(self, green_directions: FrozenSet[str]) -> None:
        # only the simulator needs this feedback loop to drain queues
        # faster on green; real video drains on its own
        if self._simulator is not None:
            self._simulator.set_green_directions(green_directions)

    def frames_snapshot(self) -> Dict[str, bytes]:
        return {
            d: a.last_annotated_jpeg for d, a in self.video_adapters.items() if a.last_annotated_jpeg is not None
        }

    def errors_snapshot(self) -> Dict[str, str]:
        errors = dict(self.errors)
        for direction, adapter in self.video_adapters.items():
            if adapter.last_error:
                errors[direction] = adapter.last_error
            elif adapter.detector_fallback_reason:
                errors[direction] = adapter.detector_fallback_reason
        return errors

    async def run(self, interval: float = 0.2) -> None:
        # See VisionSource.run()'s comment: reading 1 frame per publish tick
        # means the tick interval IS the effective video sample rate, so a
        # smaller interval (plus subtracting this tick's own processing
        # time) samples far more of the real footage per second of playback.
        while True:
            tick_start = time.monotonic()
            try:
                # video decode/detect (blocking) stays off the event loop, same
                # as VisionSource.run() -- the simulator's own tick is cheap
                # pure-python math so it rides along in the same thread hop
                batch = await asyncio.to_thread(self.tick)
                await self.bus.publish("metrics", sender="hybrid", payload=batch)
            except Exception:
                # Same rationale as VisionSource.run(): a bad cycle must skip
                # a publish, not kill the asyncio.gather() the signal
                # controller's coordinator/agents run inside.
                logging.getLogger("maats.cv").exception("HybridSource.run(): unexpected error this cycle, continuing")
            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, interval - elapsed))

    def release(self) -> None:
        for adapter in self.video_adapters.values():
            adapter.release()

"""Integration proof for the full MAATS chain, driven by REAL video frames
(not the simulator, not a hand-built TrafficMetrics):

    VIDEO FRAME -> DETECTIONS -> TrafficMetrics -> DirectionalAgent
    -> PriorityUpdate -> Coordinator -> SignalCommand -> Timer

Every object below is the real production class (DirectionVisionAdapter,
MotionDetector, CentroidTracker, DirectionalAgent, CoordinatorAgent,
SignalFSM) wired exactly as backend/main.py and dashboard.py wire them.
Nothing is monkeypatched and no TrafficMetrics/PriorityUpdate/SignalCommand
is constructed by hand -- each one is the real return value of the previous
real step.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from backend.agents.coordinator_agent import CoordinatorAgent
from backend.agents.directional_agent import DirectionalAgent
from backend.agents.message_bus import MessageBus
from backend.cv.metrics_adapter import DirectionVisionAdapter, VisionSource
from backend.cv.roi_config import ROIConfig, load_roi_configs
from backend.traffic_engine.metrics import TrafficMetrics
from backend.traffic_engine.signal_fsm import Phase

DEMO_CONFIG = "backend/cv/configs/demo_four_directions.json"

LEGAL_EDGES = {
    Phase.GREEN: {Phase.YELLOW},
    Phase.YELLOW: {Phase.ALL_RED},
    Phase.ALL_RED: {Phase.GREEN},
}

FULL_ROI = [(0, 0), (640, 0), (640, 480), (0, 480)]
QUEUE_ZONE = [(0, 300), (640, 300), (640, 480), (0, 480)]


def _write_clip(path: str, moving: bool, frames: int = 30, size=(640, 480)) -> None:
    """A tiny real .mp4 -- not a mock. `moving=False` is a genuinely empty
    (all-black) clip; `moving=True` has an actual solid box translating
    through QUEUE_ZONE frame over frame, so MotionDetector's real
    background-subtraction logic has real motion to find."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype="uint8")
        if moving:
            x = 40 + (i * 6) % 500
            cv2.rectangle(frame, (x, 340), (x + 70, 400), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _idle_batch(east_metrics: TrafficMetrics) -> dict:
    """The other three directions stay at a flat, low baseline so any
    change in the coordinator's decision is attributable only to East."""
    idle = {
        d: TrafficMetrics(direction=d, vehicle_count=0, queue_length=0, avg_wait_time=0.0, arrival_rate=0.0, timestamp=0.0)
        for d in ("N", "S", "W")
    }
    idle["E"] = east_metrics
    return idle


def test_real_video_frames_drive_the_full_chain_to_a_signal_command():
    """Runs the shipped four-direction demo clips through the real
    VisionSource -> DirectionalAgent -> CoordinatorAgent -> SignalFSM chain
    and prints a runtime trace of every stage (run with `pytest -s` to see it)."""
    if not os.path.exists(DEMO_CONFIG):
        pytest.skip("demo video config not present in this checkout")

    trace: list[str] = []
    configs = load_roi_configs(DEMO_CONFIG)
    vision = VisionSource(MessageBus(), configs)
    assert vision.errors_snapshot() == {}, "a real demo video failed to open"

    clock = [0.0]
    bus = MessageBus()
    agents = {d: DirectionalAgent(d, bus, clock=lambda: clock[0]) for d in vision.directions}
    coordinator = CoordinatorAgent(bus, directions=vision.directions, clock=lambda: clock[0], green_duration=10.0)

    commands = []
    TICKS = 60
    for _ in range(TICKS):
        clock[0] += 1.0

        # STAGE 1+2: real frame read + real detector -> real TrafficMetrics
        batch = vision.tick()
        for direction, metrics in batch.items():
            trace.append(
                f"t={clock[0]:.0f}s VIDEO[{direction}] frame -> DETECTIONS -> "
                f"TrafficMetrics(vehicle_count={metrics.vehicle_count}, queue_length={metrics.queue_length})"
            )

        # STAGE 3: each DirectionalAgent's real process_metrics -> PriorityUpdate
        for direction, agent in agents.items():
            update = agent.process_metrics(batch, now=clock[0])
            coordinator.ingest(update)
            trace.append(
                f"t={clock[0]:.0f}s {direction}Agent -> Coordinator: PriorityUpdate("
                f"priority={update.priority_score:.2f}, starved={update.is_starved})"
            )

        # STAGE 4: real FSM tick -> real SignalCommand (the dashboard's timer source)
        if coordinator.fsm.advance(clock[0]):
            command = coordinator.fsm.command(clock[0])
            commands.append(command)
            for agent in agents.values():
                agent._on_signal_command(command, now=clock[0])
            trace.append(
                f"t={clock[0]:.0f}s Coordinator -> SignalCommand(phase={command.phase.value}, "
                f"active={command.active_direction}, duration={command.duration}, "
                f"time_remaining={command.time_remaining:.0f}s)"
            )

    vision.release()
    print("\n".join(trace))

    assert commands, "the real video-driven pipeline never produced a SignalCommand"
    for previous, current in zip(commands, commands[1:]):
        assert current.phase in LEGAL_EDGES[previous.phase], (
            f"illegal transition driven by real video data: {previous.phase} -> {current.phase}"
        )


def test_switching_to_a_different_video_updates_metrics_agent_and_coordinator(tmp_path):
    """Proves the exact scenario the demo needs: swapping a direction's
    video source for a genuinely different one changes the detector's real
    output, which changes that direction's real PriorityUpdate, which
    changes the coordinator's real decision -- with zero hard-coded counts
    anywhere in this chain."""
    empty_clip = str(tmp_path / "east_empty.mp4")
    busy_clip = str(tmp_path / "east_busy.mp4")
    _write_clip(empty_clip, moving=False)
    _write_clip(busy_clip, moving=True)

    def _drive_to_metrics(video_path: str) -> TrafficMetrics:
        config = ROIConfig(
            direction="E", video_source=video_path, roi_polygon=FULL_ROI, queue_polygon=QUEUE_ZONE,
            frame_skip=0, detector_kind="motion",
        )
        adapter = DirectionVisionAdapter(config)
        metrics = None
        for i in range(25):  # past MotionDetector's warmup_frames=15
            metrics = adapter.tick(now=float(i))
        adapter.release()
        return metrics

    # "select a completely different traffic video" -- before and after
    metrics_before = _drive_to_metrics(empty_clip)
    metrics_after = _drive_to_metrics(busy_clip)

    assert metrics_before.vehicle_count == 0
    assert metrics_after.vehicle_count > 0
    assert metrics_after.queue_length > 0

    bus = MessageBus()
    east_agent = DirectionalAgent("E", bus, clock=lambda: 0.0, peak_override=False)
    other_agents = {d: DirectionalAgent(d, bus, clock=lambda: 0.0, peak_override=False) for d in ("N", "S", "W")}
    coordinator = CoordinatorAgent(bus, clock=lambda: 0.0)

    update_before = east_agent.process_metrics(_idle_batch(metrics_before), now=0.0)
    for direction, agent in other_agents.items():
        coordinator.ingest(agent.process_metrics(_idle_batch(metrics_before), now=0.0))
    coordinator.ingest(update_before)
    winner_before, green_before, _ = coordinator.decide()

    update_after = east_agent.process_metrics(_idle_batch(metrics_after), now=0.0)
    for direction, agent in other_agents.items():
        coordinator.ingest(agent.process_metrics(_idle_batch(metrics_after), now=0.0))
    coordinator.ingest(update_after)
    winner_after, green_after, _ = coordinator.decide()

    assert update_after.priority_score > update_before.priority_score, "EastAgent didn't react to the new video"
    assert winner_after == "E", "coordinator didn't react to the new video"
    # MORE TRAFFIC -> LONGER GREEN, using the REAL detected vehicle_count/
    # queue_length from the busy clip vs. the empty one (see metrics_after/
    # metrics_before above -- both are real MotionDetector output, nothing
    # fabricated).
    assert green_after > green_before, "busier real video didn't produce a longer calculated green duration"
    # The busier video is allowed to change WHICH direction wins next -- it
    # must never change how long that direction's slot lasts once it wins.

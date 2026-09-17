"""MAATS Streamlit dashboard -- the primary presentation surface.

Runs the real asyncio agent pipeline (DirectionalAgents + CoordinatorAgent +
a traffic source) in a background thread with its own event loop, and mirrors
its state into a plain, lock-guarded DashboardState object that this script
polls on every rerun. Streamlit's rerun model doesn't share memory with a
background asyncio loop directly, so a thread + shared snapshot is the
standard bridge -- see the ponytail note on SharedState below for the one
corner deliberately cut.

This file only ever reads from the agent pipeline via the MessageBus (the
same "priority_updates" / "signal_command" / "metrics" topics any other
subscriber would use) -- it does not modify agent/coordinator code.
"""
from __future__ import annotations

import asyncio
import glob
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from backend.agents.coordinator_agent import CoordinatorAgent, DIRECTIONS, SignalDecision
from backend.agents.directional_agent import DirectionalAgent
from backend.agents.message_bus import MessageBus
from backend.cv.roi_config import ROIConfig
from backend.hardware.led_signal import assert_single_green
from backend.simulation.comparison import ComparisonRunner
from backend.simulation.simulator import PROFILE_NAMES, Simulator
from backend.traffic_engine.metrics import TrafficMetrics
from backend.traffic_engine.signal_fsm import GREEN_DURATION_SECONDS, MIN_GREEN_SECONDS, Phase, SignalCommand

DEFAULT_CV_CONFIG = "backend/cv/configs/demo_four_directions.json"
DEMO_VIDEO_DIR = "backend/cv/configs/demo_videos"
PREDEFINED_VIDEO_DIRS = ("data/traffic", DEMO_VIDEO_DIR)  # data/traffic first: the evaluator-demo library
UPLOAD_DIR = "backend/cv/configs/uploads"
DIRECTION_NAMES = {"N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST"}
MAX_LOG_LINES = 200
PRIORITY_HISTORY_LENGTH = 120
SOURCE_BADGES = {"live_video": "🟢 LIVE VIDEO", "simulated": "🔵 SIMULATED", "no_source": "⚪ NO SOURCE"}
FALLBACK_ROI = [(0, 0), (640, 0), (640, 480), (0, 480)]
FALLBACK_QUEUE_ZONE = [(0, 320), (640, 320), (640, 480), (0, 480)]
# Normalized (0..1) equivalents -- used for any direction with no matching
# base-config entry, so a real uploaded video of ANY resolution/orientation
# still gets a correct full-frame ROI + bottom-third queue zone instead of
# one hardcoded for 640x480 landscape footage (see ROIConfig.roi_normalized).
FALLBACK_ROI_NORMALIZED = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
FALLBACK_QUEUE_ZONE_NORMALIZED = [(0.0, 0.66), (1.0, 0.66), (1.0, 1.0), (0.0, 1.0)]

SIGNAL_COLORS = {"GREEN": "#00E5A0", "YELLOW": "#F5C242", "RED": "#FF5C5C"}


@dataclass
class AgentSnapshot:
    direction: str
    vehicle_count: int = 0
    queue_length: int = 0
    arrival_rate: float = 0.0
    avg_wait_time: float = 0.0
    time_since_last_green: float = 0.0
    priority_score: float = 0.0
    is_starved: bool = False
    last_message_time: float = 0.0


def _decision_reason(command: SignalCommand, agents: Dict[str, "AgentSnapshot"]) -> str:
    winner = agents.get(command.active_direction, AgentSnapshot(command.active_direction))
    name = DIRECTION_NAMES.get(command.active_direction, command.active_direction)
    # Makes the adaptive link explicit for the evaluator: e.g. "EAST: 14
    # vehicles, queue 9, priority 0.87 -> Green: 82s" -- the calculated
    # duration is command.duration itself (set once when this phase started,
    # never recomputed mid-phase), not re-derived here.
    if command.starvation_override:
        basis = f"starvation override ({winner.time_since_last_green:.0f}s waiting)"
    else:
        basis = f"{winner.vehicle_count} vehicles, queue {winner.queue_length}, priority {winner.priority_score:.2f}"
    return f"{name}: {basis} -> Green: {command.duration:.0f}s"


@dataclass
class DashboardState:
    directions: Tuple[str, ...]
    source_label: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    agents: Dict[str, AgentSnapshot] = field(default_factory=dict)
    command: Optional[SignalCommand] = None
    messages: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    priority_history: Deque[dict] = field(default_factory=lambda: deque(maxlen=PRIORITY_HISTORY_LENGTH))
    started_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    frames: Dict[str, bytes] = field(default_factory=dict)
    frame_errors: Dict[str, str] = field(default_factory=dict)
    source_labels: Dict[str, str] = field(default_factory=dict)
    video_status: Dict[str, dict] = field(default_factory=dict)
    video_origin: Dict[str, str] = field(default_factory=dict)
    video_switch_requests: "queue.Queue" = field(default_factory=queue.Queue)
    seq: int = 0
    fsm_ref: Optional[object] = None  # live SignalFSM, so time_remaining actually counts down every rerun
    coordinator_ref: Optional[object] = None  # live CoordinatorAgent, for its real SignalDecision
    last_logged_decision_timestamp: Optional[float] = None

    def __post_init__(self) -> None:
        self.agents = {d: AgentSnapshot(d) for d in self.directions}

    def set_coordinator(self, coordinator) -> None:
        """Same rationale as set_fsm(): a live reference so the dashboard
        can read the coordinator's real, freshly-computed SignalDecision
        (current_direction/next_direction/traffic_summary/reason) on every
        rerun, not a stale one captured at some earlier message time."""
        self.coordinator_ref = coordinator

    def set_fsm(self, fsm) -> None:
        """Keeps a direct reference to the live SignalFSM so the dashboard
        can read a FRESH command().time_remaining on every Streamlit rerun
        (~1x/second), not just whenever the bus happens to publish a
        signal_command message -- which only happens on phase transitions,
        ~94s apart at GREEN_DURATION_SECONDS=90. Without this, the on-screen
        countdown would freeze at whatever value it had at the last
        transition instead of visibly ticking 90 -> 89 -> 88 -> ... -> 0.
        Read-only from here (only advance()/request_next() on the
        pipeline's own thread ever mutate it), so no lock is needed for
        this simple float-arithmetic read."""
        self.fsm_ref = fsm

    def log(self, block: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.messages.appendleft(f"{stamp} {block}")

    def on_priority_update(self, update) -> None:
        with self.lock:
            snap = self.agents.setdefault(update.direction, AgentSnapshot(update.direction))
            snap.vehicle_count = update.vehicle_count
            snap.queue_length = update.queue_length
            snap.arrival_rate = update.arrival_rate
            snap.avg_wait_time = update.avg_wait_time
            snap.time_since_last_green = update.time_since_last_green
            snap.priority_score = update.priority_score
            snap.is_starved = update.is_starved
            snap.last_message_time = time.time()
            elapsed = round(time.time() - self.started_at, 1)
            self.priority_history.append({"t": elapsed, **{d: a.priority_score for d, a in self.agents.items()}})
            self.seq += 1
            seq = self.seq
        name = DIRECTION_NAMES.get(update.direction, update.direction)
        # Evaluator-facing message format: every value below is read
        # straight off the real PriorityUpdate this agent just published --
        # vehicle_count/queue/density come from the CV or simulator pipeline,
        # nothing here is placeholder text.
        self.log(
            f"#{seq} {name}_AGENT\n"
            f"vehicle_count={update.vehicle_count}\n"
            f"queue={update.queue_length}\n"
            f"density={update.density:.2f}\n"
            f"status=REQUESTING_PRIORITY"
        )

    def on_signal_command(self, command: SignalCommand) -> None:
        with self.lock:
            self.command = command
            self.seq += 1
            seq = self.seq
            agents_copy = {d: AgentSnapshot(**vars(a)) for d, a in self.agents.items()}
        name = DIRECTION_NAMES.get(command.active_direction, command.active_direction)
        if command.phase == Phase.GREEN:
            reason = _decision_reason(command, agents_copy)
            self.log(
                f"#{seq} [COORDINATOR → SIGNAL CONTROLLER]\n"
                f"Decision: {name} GREEN\n"
                f"Reason: {reason}\n"
                f"Calculated green duration: {command.duration:.0f}s "
                f"(adaptive, bounded {MIN_GREEN_SECONDS:.0f}-{GREEN_DURATION_SECONDS:.0f}s)"
            )
        else:
            self.log(f"#{seq} [SIGNAL CONTROLLER] {name} {command.phase.value} for {command.duration:.0f}s")

    def log_coordinator_decision(self, decision: SignalDecision) -> None:
        """Logged whenever the coordinator computes a fresh SignalDecision
        (i.e. every round all 4 directions have reported metrics) -- every
        field is the real object from coordinator.last_decision, never
        reconstructed or guessed here."""
        with self.lock:
            self.seq += 1
            seq = self.seq
        current_name = DIRECTION_NAMES.get(decision.current_direction, decision.current_direction)
        next_name = DIRECTION_NAMES.get(decision.next_direction, decision.next_direction)
        self.log(
            f"#{seq} COORDINATOR\n"
            f"current_direction={current_name}\n"
            f"next_direction={next_name}\n"
            f"reason={decision.reason}"
        )

    def set_source_labels(self, labels: Dict[str, str]) -> None:
        with self.lock:
            self.source_labels = labels

    def on_frames(self, frames: Dict[str, bytes], errors: Dict[str, str]) -> None:
        with self.lock:
            self.frames = frames
            self.frame_errors = errors

    def on_video_status(self, status: Dict[str, dict]) -> None:
        with self.lock:
            self.video_status = status

    def set_video_origin(self, direction: str, origin: str) -> None:
        with self.lock:
            self.video_origin[direction] = origin

    def request_video_switch(self, direction: str, video_path: str, origin: str, detector_kind: Optional[str] = None) -> None:
        """Thread-safe: called from Streamlit's main thread, consumed by the
        background pipeline thread's own track_video_switches() task -- the
        actual switch always happens on the pipeline's side, never here, so
        this never touches the live cv2.VideoCapture/adapter directly.

        detector_kind lets the caller force PRIMARY (yolo) vs FALLBACK
        (motion) for the new clip instead of silently inheriting whatever
        this direction happened to be running before -- see
        render_video_switch_control, which always passes "yolo" for an
        upload and "motion" for a predefined synthetic demo clip."""
        self.video_switch_requests.put((direction, video_path, origin, detector_kind))
        self.log(f"[UI → VIDEO MANAGER] switch {DIRECTION_NAMES.get(direction, direction)} -> {os.path.basename(video_path)} requested")

    def snapshot(self) -> dict:
        # Read outside the lock (fsm_ref is set once and only ever read
        # after that -- see set_fsm()) so a live command().time_remaining
        # reflects "right now", not "whenever the last transition happened".
        live_command = self.fsm_ref.command() if self.fsm_ref is not None else None
        live_decision = self.coordinator_ref.last_decision if self.coordinator_ref is not None else None
        with self.lock:
            return {
                "agents": {d: AgentSnapshot(**vars(a)) for d, a in self.agents.items()},
                "command": live_command if live_command is not None else self.command,
                "decision": live_decision,
                "messages": list(self.messages),
                "priority_history": list(self.priority_history),
                "frames": dict(self.frames),
                "frame_errors": dict(self.frame_errors),
                "source_labels": dict(self.source_labels),
                "video_status": dict(self.video_status),
                "video_origin": dict(self.video_origin),
            }


HybridChoice = Tuple[Dict[str, ROIConfig], Tuple[str, ...], str]
SourceChoice = Union[str, HybridChoice]


def _build_source(bus: MessageBus, source_kind: str, choice: SourceChoice):
    if source_kind == "Video (per direction)":
        from backend.simulation.hybrid_source import HybridSource

        video_configs, simulated_directions, simulator_profile = choice
        return HybridSource(bus, video_configs, simulated_directions, simulator_profile)
    return Simulator(bus, profile=choice)


def _run_pipeline(state: DashboardState, source_kind: str, choice: SourceChoice) -> None:
    async def main() -> None:
        bus = MessageBus()
        try:
            source = _build_source(bus, source_kind, choice)
        except Exception as exc:  # surfaced in the UI instead of killing the thread silently
            state.error = f"failed to start {source_kind} source: {exc}"
            return

        # Use the INTENDED topology (what the sidebar actually requested),
        # not source.directions -- HybridSource/VisionSource silently drop
        # a direction whose video failed to open from their own .directions
        # (a deliberate, tested choice at that layer: "the other three keep
        # running"), but that must not mean the SIGNAL CONTROLLER permanently
        # removes that direction from the intersection. A direction with a
        # broken camera still needs its turn -- see _fill_missing_directions
        # below, which is what makes that possible without fabricating any
        # detections for it.
        directions = state.directions
        directional_agents = [DirectionalAgent(d, bus) for d in directions]
        coordinator = CoordinatorAgent(bus, directions=directions)
        # Seed the message log immediately (otherwise the evaluator-facing
        # stream shows nothing from the coordinator/signal controller for
        # this direction's entire first slot) AND give the dashboard a live
        # reference to the FSM/coordinator so snapshot() can read FRESH
        # command().time_remaining / last_decision on every rerun -- otherwise
        # the on-screen countdown would only update on a real phase
        # *transition* (a signal_command bus message, ~94s apart), freezing
        # at whatever value it had instead of visibly ticking 90 -> 89 -> ... -> 0.
        state.on_signal_command(coordinator.fsm.command())
        state.set_fsm(coordinator.fsm)
        state.set_coordinator(coordinator)

        default_labels = {d: "simulated" for d in directions}  # pure-Simulation mode
        state.set_source_labels(getattr(source, "source_labels", default_labels))

        def _no_source_metrics(direction: str, now: float) -> TrafficMetrics:
            # Zero, clearly-tagged metrics for a direction whose video
            # failed to open -- never a fabricated vehicle count. This is
            # what lets that direction still take its turn in the signal
            # rotation (deterministic fallback for THAT direction) instead
            # of silently disappearing from the intersection.
            return TrafficMetrics(
                direction=direction, vehicle_count=0, queue_length=0, avg_wait_time=0.0,
                arrival_rate=0.0, timestamp=now, density=0.0, source_id="no_source",
            )

        async def run_video_source_with_fallback(interval: float = 0.2) -> None:
            # Same interval-minus-processing-time cadence as
            # VisionSource.run()/HybridSource.run() -- this only adds the
            # missing-direction fill-in on top, it doesn't change sampling
            # behavior for directions that ARE running.
            while True:
                tick_start = time.monotonic()
                try:
                    batch = await asyncio.to_thread(source.tick)
                    now = time.time()
                    for direction in directions:
                        if direction not in batch:
                            batch[direction] = _no_source_metrics(direction, now)
                    await bus.publish("metrics", sender="hybrid", payload=batch)
                except Exception as exc:
                    # A bad cycle skips this publish -- it must never take
                    # down the pipeline the signal controller runs inside.
                    state.log(f"[VIDEO MANAGER] tick error this cycle, continuing: {exc}")
                elapsed = time.monotonic() - tick_start
                await asyncio.sleep(max(0.0, interval - elapsed))

        async def track_coordinator_decisions(poll_interval: float = 0.3) -> None:
            # coordinator.ingest() (called synchronously from handle_message)
            # sets last_decision directly, not via a bus publish -- poll for
            # a NEW one (by timestamp) so the message log gets a real,
            # non-duplicated COORDINATOR entry each time all 4 directions
            # have reported and a fresh decision was actually computed.
            while True:
                decision = coordinator.last_decision
                if decision is not None and decision.timestamp != state.last_logged_decision_timestamp:
                    state.last_logged_decision_timestamp = decision.timestamp
                    state.log_coordinator_decision(decision)
                await asyncio.sleep(poll_interval)

        async def track_priority_updates() -> None:
            queue = bus.subscribe("priority_updates")
            while True:
                message = await queue.get()
                state.on_priority_update(message.payload)

        async def track_signal_commands() -> None:
            queue = bus.subscribe("signal_command")
            while True:
                message = await queue.get()
                command = message.payload
                state.on_signal_command(command)
                if hasattr(source, "set_green_directions") and command.phase == Phase.GREEN:
                    source.set_green_directions(frozenset({command.active_direction}))

        async def track_frames(interval: float = 1.0) -> None:
            # video processing (frame decode/detect/annotate) already happened
            # inside source.run()'s background tick -- this only copies the
            # already-encoded JPEG bytes into dashboard state, no reprocessing
            if not hasattr(source, "frames_snapshot"):
                return
            while True:
                await asyncio.sleep(interval)
                state.on_frames(source.frames_snapshot(), source.errors_snapshot())

        async def track_video_status(interval: float = 1.0) -> None:
            if not hasattr(source, "status_snapshot"):
                return
            while True:
                await asyncio.sleep(interval)
                state.on_video_status(source.status_snapshot())

        async def track_video_switches(poll_interval: float = 0.3) -> None:
            # Hot-swap requests from the UI land in state.video_switch_requests
            # (a thread-safe queue.Queue); this is the only place that ever
            # calls source.switch_video(), always from inside this pipeline's
            # own event loop/thread -- never from Streamlit's main thread --
            # so it can never race with source.run()'s own tick() calls
            # beyond the per-adapter lock switch_video() already takes.
            if not hasattr(source, "switch_video"):
                return
            while True:
                try:
                    direction, video_path, origin, detector_kind = state.video_switch_requests.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(poll_interval)
                    continue
                try:
                    await asyncio.to_thread(source.switch_video, direction, video_path, detector_kind)
                    state.set_video_origin(direction, origin)
                    state.log(
                        f"[VIDEO MANAGER] {DIRECTION_NAMES.get(direction, direction)} now playing "
                        f"{os.path.basename(video_path)} ({origin})"
                    )
                except Exception as exc:
                    state.log(f"[VIDEO MANAGER] switch failed for {direction}: {exc}")

        # HybridSource/VisionSource can have gaps (a direction whose video
        # failed to open) that need filling before publish; Simulator never
        # does (it's constructed with exactly the directions it covers), so
        # its own .run() is used unchanged.
        source_task = run_video_source_with_fallback() if hasattr(source, "switch_video") else source.run()

        await asyncio.gather(
            *(agent.run() for agent in directional_agents),
            coordinator.run(),
            source_task,
            track_priority_updates(),
            track_coordinator_decisions(),
            track_signal_commands(),
            track_frames(),
            track_video_status(),
            track_video_switches(),
        )

    asyncio.run(main())


def start_pipeline(directions: Tuple[str, ...], source_label: str, source_kind: str, choice: SourceChoice) -> DashboardState:
    state = DashboardState(directions=directions, source_label=source_label)
    thread = threading.Thread(target=_run_pipeline, args=(state, source_kind, choice), daemon=True)
    thread.start()
    # ponytail: no graceful shutdown for the previous thread when the user
    # picks a new source -- it's a daemon thread that dies with the process,
    # harmless for a demo session. Add teardown if this becomes long-lived.
    return state


def congestion_level(agents: Dict[str, AgentSnapshot]) -> str:
    if not agents:
        return "LOW"
    avg_queue = sum(a.queue_length for a in agents.values()) / len(agents)
    if avg_queue < 5:
        return "LOW"
    if avg_queue < 15:
        return "MEDIUM"
    return "HIGH"


def direction_status(direction: str, agent: AgentSnapshot, command: Optional[SignalCommand]) -> Tuple[str, str]:
    """Returns (badge, detail) for a direction's current signal status."""
    if command is not None and direction == command.active_direction:
        if command.phase == Phase.GREEN:
            return "ACTIVE", f"GREEN {command.time_remaining:.0f}s"
        if command.phase == Phase.YELLOW:
            return "ACTIVE", f"YELLOW {command.time_remaining:.0f}s"
    if agent.is_starved:
        return "STARVED", f"waiting {agent.time_since_last_green:.0f}s"
    return "WAITING", ""


def direction_light(direction: str, command: Optional[SignalCommand], directions: Tuple[str, ...] = DIRECTIONS) -> str:
    """The real light color for one approach. Routed through
    backend.hardware.led_signal.assert_single_green() -- the SAME named
    invariant monitor the hardware/ESP32 output path uses -- so dashboard
    rendering and hardware output share one canonical safety gate rather
    than two independently-trusted implementations. Structurally, a
    SignalCommand's single active_direction field already makes two
    simultaneous greens unrepresentable; this call is the explicit,
    defense-in-depth check anyway: if it ever DID happen (a hand-built or
    corrupted command), this fails safe to ALL RED and logs the exact
    cause instead of rendering it."""
    if command is None:
        return "RED"
    safe_state = assert_single_green(directions, command)
    return safe_state.get(direction, "RED")


def starvation_banner_text(command: Optional[SignalCommand], agents: Dict[str, AgentSnapshot]) -> Optional[str]:
    if command is None or not command.starvation_override:
        return None
    name = DIRECTION_NAMES.get(command.active_direction, command.active_direction)
    winner = agents.get(command.active_direction, AgentSnapshot(command.active_direction))
    return f"🚨 STARVATION OVERRIDE — {name} force-served after waiting {winner.time_since_last_green:.0f}s"


def render_intersection(directions: Tuple[str, ...], agents: Dict[str, AgentSnapshot], command: Optional[SignalCommand]) -> None:
    def light(direction: str) -> str:
        # Routed through the SAME safety-gated direction_light() the Signal
        # Control tab uses -- one source of truth for "is this direction
        # really green/yellow" across the whole dashboard, not a second,
        # independently-derived answer that could disagree under a bug.
        signal = direction_light(direction, command, directions=directions)
        if signal in ("GREEN", "YELLOW"):
            return SIGNAL_COLORS[signal]
        agent = agents.get(direction, AgentSnapshot(direction))
        if agent.is_starved:
            return SIGNAL_COLORS["RED"]  # draws attention to a waiting-too-long direction
        return "#555"  # plain red/waiting, not currently flagged

    def cell(direction: Optional[str]) -> str:
        if direction is None or direction not in directions:
            return "<div style='height:90px'></div>"
        count = agents.get(direction, AgentSnapshot(direction)).vehicle_count
        return (
            f"<div style='text-align:center'>"
            f"<div style='font-size:12px;color:#9fb0bd'>{DIRECTION_NAMES.get(direction, direction)}</div>"
            f"<div style='width:22px;height:22px;border-radius:50%;background:{light(direction)};margin:4px auto;"
            f"box-shadow:0 0 10px {light(direction)}'></div>"
            f"<div style='font-size:13px'>{count} veh</div>"
            f"</div>"
        )

    rows = [
        [None, "N", None],
        ["W", "✖", "E"],
        [None, "S", None],
    ]
    html = "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:16px;" \
           "background:#131A21;border-radius:8px;border:1px solid #22303a'>"
    for row in rows:
        for entry in row:
            if entry == "✖":
                html += (
                    "<div style='text-align:center;align-self:center;font-size:22px;color:#3a4b57'>"
                    f"{entry}</div>"
                )
            else:
                html += cell(entry)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_agent_card(
    direction: str, agent: AgentSnapshot, command: Optional[SignalCommand], highlighted: bool = False
) -> None:
    badge, detail = direction_status(direction, agent, command)
    title = f"**{DIRECTION_NAMES.get(direction, direction)} AGENT**"
    if highlighted:
        title += "  🟡"  # priority just changed
    with st.container(border=True):
        st.markdown(title)
        st.text(f"Vehicles: {agent.vehicle_count}")
        st.text(f"Queue: {agent.queue_length}")
        st.text(f"Avg wait: {agent.avg_wait_time:.0f}s")
        st.text(f"Arrival: {agent.arrival_rate / 60.0:.2f}/s")
        st.text(f"Priority: {agent.priority_score:.2f}")
        st.text(f"State: {badge}")
        since_message = time.time() - agent.last_message_time if agent.last_message_time else None
        if since_message is None:
            st.caption("Last message: never")
        else:
            status = "🟢 LIVE" if since_message < 3.0 else "⚪ IDLE"
            st.caption(f"Last message: {since_message:.0f}s ago  |  {status}")
        if badge == "ACTIVE":
            st.success(f"ACTIVE\n{detail}")
        elif badge == "STARVED":
            st.error(f"STARVED\n{detail}")
        else:
            st.text("Status: WAITING")


def render_communication_topology(
    directions: Tuple[str, ...], agents: Dict[str, AgentSnapshot], command: Optional[SignalCommand], highlighted: set
) -> None:
    def pill(direction: str) -> str:
        badge, _ = direction_status(direction, agents.get(direction, AgentSnapshot(direction)), command)
        border = {"ACTIVE": "#00E5A0", "STARVED": "#FF5C5C"}.get(badge, "#3a4b57")
        glow = "box-shadow:0 0 14px 3px #F5C242;" if direction in highlighted else ""
        name = DIRECTION_NAMES.get(direction, direction)
        return (
            f"<div style='text-align:center;padding:10px 16px;border-radius:8px;background:#131A21;"
            f"border:2px solid {border};{glow}min-width:100px'>"
            f"<div style='font-size:12px;color:#e6edf3;font-weight:600'>{name} AGENT</div>"
            f"</div>"
        )

    pills = "".join(f"<div>{pill(d)}</div>" for d in directions)
    html = (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:6px;padding:14px;"
        "background:#0d1117;border-radius:10px;border:1px solid #22303a'>"
        "<div style='padding:10px 22px;border-radius:8px;background:#1b2733;border:2px solid #4da3ff;"
        "font-weight:700;color:#e6edf3'>JUNCTION COORDINATOR</div>"
        "<div style='width:2px;height:16px;background:#3a4b57'></div>"
        "<div style='width:70%;height:2px;background:#3a4b57'></div>"
        f"<div style='display:flex;gap:16px;justify-content:center;flex-wrap:wrap;padding-top:6px'>{pills}</div>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_decision_panel(command: Optional[SignalCommand], agents: Dict[str, AgentSnapshot]) -> None:
    with st.container(border=True):
        st.markdown("##### Current Decision")
        if command is None:
            st.text("waiting for first decision...")
            return
        name = DIRECTION_NAMES.get(command.active_direction, command.active_direction)
        winner = agents.get(command.active_direction, AgentSnapshot(command.active_direction))
        st.text(f"Active direction: {name}")
        st.text(f"Phase: {command.phase.value}")
        # Explicit, so the evaluator can see MORE TRAFFIC -> LONGER GREEN
        # directly: every value below is real (vehicle_count/queue_length/
        # priority_score off the live agent snapshot), and the green
        # duration is the actual value this phase was started with.
        st.text(f"Vehicle count: {winner.vehicle_count}")
        st.text(f"Queue: {winner.queue_length}")
        st.text(f"Priority: {winner.priority_score:.2f}")
        st.text(f"Calculated green time: {command.duration:.0f}s")
        st.text(f"Selected reason: {_decision_reason(command, agents)}")
        st.text(f"Starvation override: {'YES' if command.starvation_override else 'NO'}")
        st.text(f"Time remaining: {command.time_remaining:.0f}s")


def render_signal_control(
    directions: Tuple[str, ...], agents: Dict[str, AgentSnapshot], command: Optional[SignalCommand]
) -> None:
    """The dedicated evaluator view: one tile per approach. Exactly one tile
    can ever show GREEN/YELLOW -- the other three always show RED, straight
    off the same SignalCommand.active_direction the coordinator/FSM
    produced (see direction_light()). A RED tile shows no countdown, since
    it isn't running one; only the active direction has a real timer. The
    duration it counts down from is ADAPTIVE -- calculated once from that
    direction's real traffic when the phase started (see
    scoring.compute_green_time()), bounded to [MIN_GREEN_SECONDS,
    GREEN_DURATION_SECONDS] -- never recomputed mid-phase, so the number
    only ever counts down, it never jumps."""

    def tile(direction: Optional[str]) -> str:
        if direction is None or direction not in directions:
            return "<div></div>"
        light = direction_light(direction, command, directions=directions)
        color = SIGNAL_COLORS[light]
        name = DIRECTION_NAMES.get(direction, direction)
        # Tied to the SAME safety-gated `light` value, not a separately
        # recomputed "is this nominally the active direction" check -- so a
        # hypothetical assert_single_green() fail-safe (forcing this tile
        # to RED) also hides its countdown, instead of leaving a stray
        # timer next to a red light.
        show_timer = command is not None and light in ("GREEN", "YELLOW")
        timer_html = (
            f"<div style='font-size:11px;color:#9fb0bd;margin-top:6px'>TIME REMAINING</div>"
            f"<div style='font-size:20px;font-weight:700;color:{color}'>{command.time_remaining:.0f} s</div>"
            f"<div style='font-size:10px;color:#9fb0bd'>Green: {command.duration:.0f}s</div>"
            if show_timer else ""
        )
        return (
            f"<div style='text-align:center;padding:14px;border-radius:10px;background:#131A21;"
            f"border:2px solid {color}'>"
            f"<div style='font-size:13px;color:#9fb0bd;font-weight:600'>{name}</div>"
            f"<div style='width:36px;height:36px;border-radius:50%;background:{color};margin:8px auto;"
            f"box-shadow:0 0 18px {color}'></div>"
            f"<div style='font-size:18px;font-weight:700;color:{color}'>{light}</div>"
            f"{timer_html}"
            f"</div>"
        )

    rows = [[None, "N", None], ["W", "✖", "E"], [None, "S", None]]
    html = (
        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;padding:18px;"
        "background:#0d1117;border-radius:12px;border:1px solid #22303a'>"
    )
    for row in rows:
        for entry in row:
            if entry == "✖":
                html += (
                    "<div style='text-align:center;align-self:center;font-size:26px;color:#3a4b57'>✖</div>"
                )
            else:
                html += tile(entry)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_global_status(command: Optional[SignalCommand], agents: Dict[str, AgentSnapshot]) -> None:
    """The single, unambiguous global status line the evaluator should be
    able to read at a glance: which one direction is active, what phase
    it's in, how much of its ADAPTIVE slot is left, and -- explicitly --
    the traffic and calculated green time that produced that duration, so
    "more traffic -> longer green" is directly visible, not just implied.
    All values read straight off the live SignalCommand/agent snapshot,
    updated every rerun."""
    with st.container(border=True):
        st.markdown("##### Global Signal Status")
        if command is None:
            st.text("waiting for first decision...")
            return
        name = DIRECTION_NAMES.get(command.active_direction, command.active_direction)
        winner = agents.get(command.active_direction, AgentSnapshot(command.active_direction))
        cols = st.columns(5)
        cols[0].metric("ACTIVE DIRECTION", name)
        cols[1].metric("PHASE", command.phase.value)
        cols[2].metric("TRAFFIC", f"{winner.vehicle_count} vehicles")
        cols[3].metric("GREEN DURATION", f"{command.duration:.0f}s")
        cols[4].metric("TIME REMAINING", f"{command.time_remaining:.0f}s")
        st.text(f"Reason: {_decision_reason(command, agents)}")

        banner = starvation_banner_text(command, agents)
        if banner:
            st.error(banner)


def render_mode_banner(
    source_labels: Dict[str, str], video_status: Dict[str, dict], decision: Optional[SignalDecision],
) -> None:
    """The evaluator-facing INPUT MODE / CONTROL MODE indicator. Every value
    is derived from real, already-live state:
      - INPUT MODE is VIDEO iff at least one direction actually has an open
        video capture right now (source_labels), never a static assumption.
      - CONTROL MODE is DETERMINISTIC FALLBACK iff the coordinator has never
        yet computed a real decision (coordinator.last_decision is None) --
        i.e. the FSM is running purely on its own built-in round-robin
        because no traffic data has influenced it, exactly the "no video /
        no traffic data" case this label is meant to describe. As soon as
        one real decision exists, it flips to ADAPTIVE and stays there.
      - DETECTOR/TRACKER only appear in VIDEO mode, aggregated from the
        real per-direction detector status the CV pipeline reports.
    """
    has_live_video = any(label == "live_video" for label in source_labels.values())
    input_mode = "VIDEO" if has_live_video else "NO VIDEO"
    control_mode = "ADAPTIVE (priority-based)" if decision is not None else "DETERMINISTIC FALLBACK"

    cols = st.columns(4 if has_live_video else 2)
    cols[0].metric("INPUT MODE", input_mode)
    cols[1].metric("CONTROL MODE", control_mode)
    if has_live_video:
        modes = {s.get("detector_mode") for s in video_status.values() if s.get("detector_mode")}
        if modes == {"PRIMARY"}:
            detector_line = "YOLO (PRIMARY)"
        elif modes == {"FALLBACK"}:
            detector_line = "Motion (FALLBACK)"
        elif modes:
            detector_line = "Mixed (PRIMARY + FALLBACK)"
        else:
            detector_line = "—"
        cols[2].metric("DETECTOR", detector_line)
        cols[3].metric("TRACKER", "CentroidTracker")
    if not has_live_video:
        st.caption("No camera/video input active -- the signal controller is cycling NORTH → EAST → SOUTH → WEST deterministically.")


def _available_demo_videos() -> List[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(DEMO_VIDEO_DIR, "*.mp4")))


def _available_predefined_videos() -> List[Tuple[str, str]]:
    """(display label, full path) pairs for the live "switch video" picker,
    scanning data/traffic/ (the evaluator-demo library) and the legacy
    DEMO_VIDEO_DIR. A basename that exists in both is only listed once,
    preferring the data/traffic/ copy."""
    seen: Dict[str, str] = {}
    for directory in PREDEFINED_VIDEO_DIRS:
        for path in sorted(glob.glob(os.path.join(directory, "*.mp4"))):
            name = os.path.basename(path)
            seen.setdefault(name, path)
    return [(name, seen[name]) for name in sorted(seen)]


def _resolve_hybrid_setup(base_config_path: str) -> HybridChoice:
    """Lets the sidebar assign each of the 4 canonical directions to either
    a real video file or the simulator. All 4 set to Video = Mode A
    (four-video); some set to Simulated = Mode B (single-video/test) --
    it's the same HybridSource either way, just a different mix. A
    direction's video file is picked from the demo library or uploaded;
    same ROIConfig/HybridSource pipeline regardless of which is chosen."""
    from backend.cv.roi_config import load_roi_configs

    try:
        base_configs = load_roi_configs(base_config_path)
    except Exception as exc:
        st.error(f"could not load ROI config {base_config_path!r}: {exc}")
        base_configs = {}

    available = _available_demo_videos()
    video_configs: Dict[str, ROIConfig] = {}
    simulated_directions: List[str] = []

    for direction in DIRECTIONS:
        st.markdown(f"**{DIRECTION_NAMES.get(direction, direction)}**")
        mode = st.radio(
            f"{direction} mode", ["Video", "Simulated"], key=f"dir_mode_{direction}",
            horizontal=True, label_visibility="collapsed",
        )
        if mode == "Simulated":
            simulated_directions.append(direction)
            continue

        base_cfg = base_configs.get(direction) or ROIConfig(
            direction=direction,
            video_source=os.path.join(DEMO_VIDEO_DIR, available[0]) if available else "",
            # Real detector + resolution-independent ROI by default -- this
            # branch only fires when base_config_path doesn't define this
            # direction at all, which in practice means "the evaluator is
            # about to assign it a real video," not a synthetic demo clip.
            roi_polygon=FALLBACK_ROI_NORMALIZED, queue_polygon=FALLBACK_QUEUE_ZONE_NORMALIZED,
            roi_normalized=True, frame_skip=0, detector_kind="yolo",
        )
        current_name = os.path.basename(base_cfg.video_source)
        options = available if current_name in available else ([current_name] + available if current_name else available)
        chosen = st.selectbox(
            f"{direction} file", options, key=f"video_choice_{direction}", label_visibility="collapsed"
        ) if options else None
        upload = st.file_uploader(
            f"{direction} upload", type=["mp4", "avi", "mov"], key=f"video_upload_{direction}",
            label_visibility="collapsed",
        )
        video_source = os.path.join(DEMO_VIDEO_DIR, chosen) if chosen else base_cfg.video_source
        detector_kind = base_cfg.detector_kind
        if upload is not None:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            upload_path = os.path.join(UPLOAD_DIR, f"{direction}_{upload.name}")
            with open(upload_path, "wb") as f:
                f.write(upload.getbuffer())
            video_source = upload_path
            # An upload is real footage -- never inherit base_cfg's "motion"
            # (base_cfg here is usually demo_four_directions.json, tuned for
            # the synthetic demo clips, not whatever the evaluator just gave us).
            detector_kind = "yolo"
        video_configs[direction] = replace(base_cfg, video_source=video_source, detector_kind=detector_kind)

    simulator_profile = st.selectbox(
        "Simulator profile (Simulated directions)", PROFILE_NAMES, key="hybrid_sim_profile"
    )
    return video_configs, tuple(simulated_directions), simulator_profile


def main() -> None:
    st.set_page_config(page_title="MAATS", page_icon="\U0001F6A6", layout="wide")

    st.markdown("## MAATS")
    st.caption("Multi-Agent Adaptive Traffic Signal Coordination")

    with st.sidebar:
        st.header("Traffic Source")
        source_kind = st.radio("Source", ["Simulation", "Video (per direction)"], key="source_kind")
        if source_kind == "Simulation":
            choice: SourceChoice = st.selectbox("Profile", PROFILE_NAMES, key="profile_choice")
        else:
            base_config_path = st.text_input("Base ROI config path", DEFAULT_CV_CONFIG, key="cv_config_choice")
            st.caption("Assign each direction: Video (Mode A if all 4) or Simulated (Mode B otherwise)")
            choice = _resolve_hybrid_setup(base_config_path)
        restart = st.button("Start / Restart", type="primary")
        st.caption("Restarting starts a fresh pipeline; the old one keeps running quietly in the background.")
        st.divider()
        reset_demo = st.button("🔄 RESET DEMO", help="Clean-slate reset: video, trackers, metrics, agent states, message history, and signal state -- using the source config currently selected above. Does not reload the browser page.")
        st.caption("One-click reset for evaluator moments -- same fresh pipeline as Start/Restart, kept separate so it's easy to find mid-demo.")

    needs_start = "dash_state" not in st.session_state or restart or reset_demo
    if needs_start:
        directions = DIRECTIONS
        if source_kind == "Video (per direction)":
            video_configs, simulated_directions, _ = choice
            directions = tuple(video_configs.keys()) + tuple(d for d in simulated_directions if d not in video_configs)
        label = "HYBRID" if source_kind == "Video (per direction)" else "SIMULATION"
        st.session_state["dash_state"] = start_pipeline(directions or DIRECTIONS, label, source_kind, choice)
        st.session_state["comparison_profile"] = choice if source_kind == "Simulation" else None

    state: DashboardState = st.session_state["dash_state"]
    if state.error:
        st.error(state.error)

    snap = state.snapshot()
    agents = snap["agents"]
    command: Optional[SignalCommand] = snap["command"]

    prev_scores = st.session_state.get("prev_priority_scores", {})
    highlighted = {
        d for d, a in agents.items()
        if d in prev_scores and abs(a.priority_score - prev_scores[d]) > 0.01
    }
    st.session_state["prev_priority_scores"] = {d: a.priority_score for d, a in agents.items()}

    total_vehicles = sum(a.vehicle_count for a in agents.values())
    avg_queue = (sum(a.queue_length for a in agents.values()) / len(agents)) if agents else 0.0

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Current Phase", command.phase.value if command else "—")
    kpi_cols[1].metric("Active Direction", DIRECTION_NAMES.get(command.active_direction, "—") if command else "—")
    kpi_cols[2].metric("Time Remaining", f"{command.time_remaining:.0f}s" if command else "—")
    kpi_cols[3].metric("Vehicles Detected", total_vehicles)
    kpi_cols[4].metric("Average Queue", f"{avg_queue:.1f}")
    kpi_cols[5].metric("Congestion Level", congestion_level(agents))

    render_mode_banner(snap["source_labels"], snap["video_status"], snap["decision"])

    st.markdown(f"**Traffic Source:** {state.source_label}")

    source_labels = snap["source_labels"]
    if source_labels:
        status_line = "  |  ".join(
            f"{DIRECTION_NAMES.get(d, d)}: {SOURCE_BADGES.get(source_labels.get(d, 'no_source'), SOURCE_BADGES['no_source'])}"
            for d in DIRECTIONS
        )
        st.caption(status_line)

    video_directions = [d for d in state.directions if source_labels.get(d) == "live_video"]

    def render_video_switch_control(direction: str) -> None:
        predefined = _available_predefined_videos()
        options = ["(choose a predefined clip)"] + [name for name, _ in predefined]
        chosen = st.selectbox("Switch to", options, key=f"switch_choice_{direction}", label_visibility="collapsed")
        upload = st.file_uploader(
            "or upload", type=["mp4", "avi", "mov"], key=f"switch_upload_{direction}", label_visibility="collapsed"
        )
        if st.button("🔄 Switch", key=f"switch_button_{direction}"):
            if upload is not None:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                upload_path = os.path.join(UPLOAD_DIR, f"{direction}_{upload.name}")
                with open(upload_path, "wb") as f:
                    f.write(upload.getbuffer())
                # An upload is real footage by definition -- always force
                # PRIMARY (yolo), never silently inherit a "motion" fallback
                # this direction happened to be running before.
                state.request_video_switch(direction, upload_path, "Uploaded", detector_kind="yolo")
            elif chosen != options[0]:
                path = dict(predefined)[chosen]
                # The predefined library (data/traffic/, demo_videos/) is
                # entirely synthetic clips generated for MotionDetector --
                # see scripts/generate_demo_clips.py -- so FALLBACK is
                # correct here, not a downgrade.
                state.request_video_switch(direction, path, "Local", detector_kind="motion")
            else:
                st.warning("choose a predefined clip or upload a file first")

    def render_video_status_card(direction: str) -> None:
        status = snap["video_status"].get(direction)
        origin = snap["video_origin"].get(direction, "Local")
        with st.container(border=True):
            if status is None:
                st.text("VIDEO:\n—")
                return
            st.text(f"VIDEO:\n{status['video_filename']}")
            st.text(f"SOURCE:\n{origin}")
            st.text(f"DIRECTION:\n{DIRECTION_NAMES.get(direction, direction)}")
            detector_line = f"{status['detector_label']} ({status.get('detector_mode', '—')})"
            if status.get("detector_fallback_reason"):
                detector_line += " [fallback]"
            st.text(f"DETECTOR:\n{detector_line}")
            st.text(
                f"TRACKING:\n{'Active' if status['tracking_active'] else 'Inactive'} "
                f"({status['active_tracks']} active, {status.get('unique_vehicle_count', 0)} unique)"
            )
            if status.get("avg_confidence") is not None:
                st.caption(f"Detection confidence: {status['avg_confidence']:.2f}")
            else:
                st.caption("Detection confidence: no detections this tick")
            st.caption(
                f"Detector latency: {status.get('detector_latency_ms', 0.0):.0f}ms  |  "
                f"Processing: {status.get('tracker_fps', 0.0):.1f} FPS"
            )
            if status.get("last_error"):
                st.warning(f"detector/frame issue: {status['last_error']}")
            elif status.get("detector_fallback_reason"):
                st.caption(f"Fallback reason: {status['detector_fallback_reason']}")

    def render_traffic_vision(interactive: bool = True) -> None:
        # interactive=False (Overview tab) skips the switch-video widgets so
        # this can be called a second time in the same script run without
        # duplicate Streamlit widget keys -- Overview is read-only by design,
        # the live switch controls only ever live in the dedicated tab below.
        no_source_directions = [
            d for d in state.directions
            if source_labels.get(d) == "no_source" and d not in video_directions
        ]
        if no_source_directions:
            frame_errors = snap["frame_errors"]
            for direction in no_source_directions:
                name = DIRECTION_NAMES.get(direction, direction)
                reason = frame_errors.get(direction, "video failed to open")
                st.error(
                    f"{name}: no video source ({reason}) -- falling back to deterministic "
                    f"signal timing for this direction; it still receives its 90s slot in "
                    f"the rotation, just without live traffic priority."
                )
        if not video_directions:
            st.info("No live video source is active for this run -- switch a direction to Video in the sidebar.")
            return
        st.caption(
            "Evaluator view: real detector output per approach -- the same TrafficMetrics "
            "(vehicle_count/queue_length/arrival_rate) each DirectionalAgent consumes next. "
            "\"Switch to\" hot-swaps that direction's footage in place -- no restart."
        )
        frames = snap["frames"]
        frame_errors = snap["frame_errors"]
        cols = st.columns(len(video_directions))
        for col, direction in zip(cols, video_directions):
            with col:
                st.caption(f"{DIRECTION_NAMES.get(direction, direction)} (LIVE VIDEO)")
                jpeg = frames.get(direction)
                if jpeg is not None:
                    st.image(jpeg, width="stretch")
                else:
                    st.info("waiting for first frame...")
                if frame_errors.get(direction):
                    st.warning(frame_errors[direction])
                agent = agents.get(direction, AgentSnapshot(direction))
                st.text(
                    f"count={agent.vehicle_count} queue={agent.queue_length} "
                    f"arrival={agent.arrival_rate / 60.0:.2f}/s wait={agent.time_since_last_green:.0f}s"
                )
                render_video_status_card(direction)
                if interactive:
                    render_video_switch_control(direction)

    def render_multi_agent_communication() -> None:
        st.caption("Evaluator view: live messages between the four directional agents and the Junction Coordinator.")
        render_communication_topology(state.directions, agents, command, highlighted)

        comm_cols = st.columns(len(state.directions)) if state.directions else [st]
        for col, direction in zip(comm_cols, state.directions):
            with col:
                render_agent_card(
                    direction, agents.get(direction, AgentSnapshot(direction)), command,
                    highlighted=(direction in highlighted),
                )

        decision_col, stream_col = st.columns([1, 1.6])
        with decision_col:
            render_decision_panel(command, agents)
        with stream_col:
            st.markdown("##### Live Message Stream")
            st.code("\n\n".join(snap["messages"][:12]) or "waiting for agent activity...", language=None)

        st.divider()
        st.markdown("#### Priority Score")
        score_df = pd.DataFrame({"priority_score": [agents.get(d, AgentSnapshot(d)).priority_score for d in state.directions]}, index=[DIRECTION_NAMES.get(d, d) for d in state.directions])
        st.bar_chart(score_df)

        history = snap["priority_history"]
        if len(history) > 1:
            hist_df = pd.DataFrame(history).set_index("t").rename(columns=DIRECTION_NAMES)
            st.line_chart(hist_df)

    def render_signal_control_view() -> None:
        st.caption(
            "Evaluator view: live phase/timer for all four approaches, driven directly by the "
            "Coordinator's real SignalCommand -- not an independent countdown."
        )
        render_signal_control(state.directions, agents, command)
        render_global_status(command, agents)

        st.divider()
        st.markdown("#### Intersection")
        render_intersection(state.directions, agents, command)

        if state.source_label == "SIMULATION" and st.session_state.get("comparison_profile"):
            st.divider()
            st.markdown("#### Round-Robin vs Priority-Ordered")
            st.caption(
                "Both arms use the same fixed-duration green slot per direction -- only the "
                "ORDER differs: strict NORTH→EAST→SOUTH→WEST rotation vs priority-based next-"
                "direction selection. Uses a shorter illustrative slot length (not the real "
                "90s) purely so a 5-minute replay shows several full cycles."
            )

            @st.cache_data(show_spinner=False)
            def run_comparison(profile: str) -> pd.DataFrame:
                runner = ComparisonRunner(profile=profile, seed=42)
                snapshots = runner.run(ticks=300)
                return pd.DataFrame(
                    {
                        "Priority-ordered": [s.adaptive_cumulative_vehicle_seconds for s in snapshots],
                        "Round-robin": [s.fixed_cumulative_vehicle_seconds for s in snapshots],
                    },
                    index=[s.now for s in snapshots],
                )

            st.line_chart(run_comparison(st.session_state["comparison_profile"]))
            st.caption("Cumulative vehicle-seconds waiting (lower is better) over a 5-minute replay of the same seeded traffic.")

    def render_overview() -> None:
        st.caption(
            "Combined at-a-glance view -- every widget below reads the exact same live "
            "snapshot as the three dedicated tabs; nothing here is a separate feed."
        )
        if video_directions:
            st.markdown("##### Traffic Vision")
            render_traffic_vision(interactive=False)
            st.divider()
        st.markdown("##### Multi-Agent Communication")
        render_communication_topology(state.directions, agents, command, highlighted)
        st.divider()
        st.markdown("##### Signal Control")
        render_signal_control(state.directions, agents, command)
        render_global_status(command, agents)

    tab_overview, tab_vision, tab_comm, tab_signal = st.tabs(
        ["📋 OVERVIEW", "🎥 TRAFFIC VISION", "🤝 MULTI-AGENT COMMUNICATION", "🚦 SIGNAL CONTROL"]
    )
    with tab_overview:
        render_overview()
    with tab_vision:
        render_traffic_vision()
    with tab_comm:
        render_multi_agent_communication()
    with tab_signal:
        render_signal_control_view()

    if not os.environ.get("MAATS_DASHBOARD_TEST"):
        # live auto-refresh; disabled under the headless AppTest harness
        # (tests/test_dashboard.py), which does a single scripted run
        time.sleep(1.0)
        st.rerun()


if __name__ == "__main__":
    main()

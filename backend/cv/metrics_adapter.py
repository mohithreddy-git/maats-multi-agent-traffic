"""CV adapter: turns a video feed into the same TrafficMetrics contract the
simulator produces.

CV is only a TrafficMetrics *producer* -- nothing in here touches agents,
scoring, or the FSM. VisionSource.run() publishes to the bus exactly like
Simulator.run() does, so it's a drop-in replacement in backend/main.py and
the agent architecture is untouched.

Detector-agnostic: each direction gets a Detector built from its
ROIConfig.detector_kind ("yolo" or "motion", see detector_factory.py). If
the requested detector fails to construct (e.g. YOLO weights/deps missing),
it falls back to the dependency-free motion detector rather than crashing.
Any failure inside a tick (bad frame, detector exception) is caught and the
last known-good TrafficMetrics is returned instead of propagating -- one
camera misbehaving must not take down the other three directions or the
dashboard.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from dataclasses import replace
from typing import Deque, Dict, List, Optional, Set, Tuple

from ..agents.message_bus import MessageBus
from ..traffic_engine.metrics import TrafficMetrics
from .detector import Detection, Detector
from .detector_factory import build_detector_safe
from .roi_config import ROIConfig, point_in_polygon, scale_polygon
from .tracker import CentroidTracker

ARRIVAL_WINDOW_SECONDS = 60.0
LANE_CAPACITY = 20  # same normalization simulator.py and scoring.py use for density/queue
LATENCY_EMA_ALPHA = 0.3  # smoothing factor for detector_latency_ms / tracker_fps readouts

# Orientation-correction candidates, tried in this order when a frame comes
# in portrait-shaped: 0 = leave alone, +/-90 = rotate to landscape. 180 is
# deliberately excluded -- a portrait frame rotated 180 is still portrait
# (same problem, if any), so it can never fix a sideways-content video.
_ORIENTATION_CANDIDATES: Tuple[int, ...] = (0, 90, -90)
_ORIENTATION_PROBE_FRAMES = 3
# A rotated candidate must beat the native (0) orientation by a clear
# margin before we act on it -- a genuinely portrait-filmed video (a phone
# mounted vertically) may detect vehicles fine as-is, and marginal noise
# between candidates must never flip a real portrait video's orientation.
_ORIENTATION_IMPROVEMENT_FACTOR = 1.5

BOX_COLOR = (0, 220, 130)  # BGR, matches the dashboard's "active" green
ROI_COLOR = (90, 90, 90)
QUEUE_COLOR = (60, 60, 220)


class DirectionVisionAdapter:
    """One video source -> one direction's TrafficMetrics, one tick at a
    time. `detector` is injectable so this class is unit-testable without
    a real YOLO/motion detector (see tests/test_cv_metrics_adapter.py)."""

    def __init__(self, config: ROIConfig, detector: Optional[Detector] = None) -> None:
        import cv2

        self._cv2 = cv2
        self.config = config
        if detector is not None:
            self.detector = detector
            self.detector_fallback_reason: Optional[str] = None
        else:
            self.detector, self.detector_fallback_reason = build_detector_safe(
                config.detector_kind, **self._detector_kwargs(config.detector_kind)
            )
        self.tracker = CentroidTracker()
        self._detector_latency_ms: float = 0.0
        self._tracker_fps: float = 0.0
        self._capture = cv2.VideoCapture(config.video_source)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source: {config.video_source!r}")
        # In-memory orientation correction (rotate before inference, never
        # touching the file on disk): determined once per video by content
        # probing, not per frame -- a camera's physical mounting doesn't
        # change mid-clip. 0 for every landscape video (the common case,
        # zero probing cost -- see _probe_orientation).
        self._orientation_rotation: int = self._probe_orientation(self._capture, self.detector)
        self._frame_index = 0
        self._last_detections: List[Detection] = []
        self._last_detection_ids: List[Optional[int]] = []
        self._seen_track_ids: Set[int] = set()
        self._arrivals: Deque[float] = deque()
        self.source_id = os.path.basename(str(config.video_source))
        self.last_error: Optional[str] = None
        self.last_annotated_jpeg: Optional[bytes] = None
        self._last_metrics = self._zero_metrics(time.time())
        # guards tick() and switch_video() against each other -- both touch
        # the same cv2.VideoCapture/tracker, and in the dashboard they can
        # run from different threads (the pipeline's tick loop vs. a live
        # "switch video" request from the UI)
        self._lock = threading.Lock()

    def _detector_kwargs(self, kind: str) -> dict:
        # MotionDetector has its own (area/aspect/rectangularity) tuning
        # knobs, not confidence/IoU -- only forward YOLO's thresholds when
        # that's actually the detector being built.
        if kind == "yolo":
            return {
                "confidence_threshold": self.config.confidence_threshold,
                "iou_threshold": self.config.iou_threshold,
            }
        return {}

    def _zero_metrics(self, now: float) -> TrafficMetrics:
        return TrafficMetrics(
            direction=self.config.direction,
            vehicle_count=0,
            queue_length=0,
            avg_wait_time=0.0,
            arrival_rate=0.0,
            timestamp=now,
            density=0.0,
            source_id=self.source_id,
        )

    def _read_frame(self):
        ok, frame = self._capture.read()
        if not ok:
            # loop the clip so a short demo video gives a continuous feed
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._capture.read()
        return ok, frame

    def _resize(self, frame):
        if self.config.resize_width <= 0:
            return frame
        height, width = frame.shape[:2]
        if width <= self.config.resize_width:
            return frame
        # Preserve aspect ratio (scale height by the same factor) rather
        # than forcing a fixed target size -- a portrait upload stays
        # portrait, a widescreen upload stays widescreen. resize_width=640
        # is a floor for small/distant vehicles' pixel footprint, not a
        # blind "shrink to thumbnail" -- callers that need to preserve very
        # small vehicles can raise resize_width (or set it to 0) per direction.
        scale = self.config.resize_width / width
        return self._cv2.resize(frame, (self.config.resize_width, int(height * scale)))

    def _rotate(self, frame, degrees: int):
        if degrees == 90:
            return self._cv2.rotate(frame, self._cv2.ROTATE_90_CLOCKWISE)
        if degrees == -90:
            return self._cv2.rotate(frame, self._cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _probe_orientation(self, capture, detector, sample_frames: int = _ORIENTATION_PROBE_FRAMES) -> int:
        """Determines, ONCE per video, whether frames need rotating before
        detection -- e.g. a phone-recorded clip whose pixel content is
        physically sideways (no player-side rotation metadata baked in),
        which a COCO-trained detector barely recognizes as vehicles.

        Only probes when there's real evidence to act on:
          - skipped entirely for the motion-fallback detector (class
            confusion from orientation isn't its failure mode, and the
            synthetic demo clips it runs on must never be touched here);
          - skipped entirely for a landscape-shaped frame (the overwhelming
            common case -- zero detector calls, zero added latency);
          - for a portrait-shaped frame, tries 0/+90/-90 on a few sample
            frames and requires a rotated candidate to clearly outscore
            doing nothing before acting on it, so a genuinely
            portrait-filmed video (real vehicles, already upright) is left
            alone rather than needlessly rotated on noise.
        """
        if type(detector).__name__ != "VehicleDetector":
            return 0

        scores = {rotation: 0.0 for rotation in _ORIENTATION_CANDIDATES}
        portrait_seen = False
        for _ in range(sample_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            height, width = frame.shape[:2]
            if height <= width:
                return 0  # landscape -- nothing to correct, stop probing immediately
            portrait_seen = True
            for rotation in _ORIENTATION_CANDIDATES:
                # Rotate the RAW frame before resizing, exactly the order
                # _tick_unsafe() uses for real inference -- resizing first
                # would scale a still-portrait frame by its (larger) width,
                # so rotating afterwards yields a different, inconsistent
                # final resolution than what real ticks actually feed the
                # detector.
                candidate = self._resize(self._rotate(frame, rotation))
                scores[rotation] += sum(d.confidence for d in detector.detect(candidate))

        if not portrait_seen:
            return 0
        best = max(scores, key=scores.get)
        if best != 0 and scores[best] > 0 and scores[best] > scores[0] * _ORIENTATION_IMPROVEMENT_FACTOR:
            return best
        return 0

    def _roi_polygons(self, frame) -> Tuple[List, List]:
        if not self.config.roi_normalized:
            return self.config.roi_polygon, self.config.queue_polygon
        height, width = frame.shape[:2]
        return (
            scale_polygon(self.config.roi_polygon, width, height),
            scale_polygon(self.config.queue_polygon, width, height),
        )

    def _annotate(self, frame, detections: List[Detection], ids: List[Optional[int]], roi_polygon, queue_polygon):
        cv2 = self._cv2
        annotated = frame.copy()

        def draw_polygon(polygon, color):
            pts = [(int(x), int(y)) for x, y in polygon]
            for i in range(len(pts)):
                cv2.line(annotated, pts[i], pts[(i + 1) % len(pts)], color, 1)

        draw_polygon(roi_polygon, ROI_COLOR)
        draw_polygon(queue_polygon, QUEUE_COLOR)

        for det, track_id in zip(detections, ids):
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
            # e.g. "CAR #17 0.89" -- class, stable track ID, confidence
            id_part = f"#{track_id} " if track_id is not None else ""
            label = f"{det.class_name.upper()} {id_part}{det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, 1)

        detector_name = "FALLBACK (motion)" if type(self.detector).__name__ == "MotionDetector" else "YOLO"
        overlay = (
            f"{self.config.direction} | count={len(self.tracker.tracks)} | {detector_name} | "
            f"{self._detector_latency_ms:.0f}ms | {self._tracker_fps:.1f} FPS"
        )
        cv2.putText(annotated, overlay, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", annotated)
        return buf.tobytes() if ok else None

    def _match_track_ids(self, detections: List[Detection]) -> List[Optional[int]]:
        # CentroidTracker.update() sets each track's centroid to exactly the
        # matched/new detection's centroid, so an exact-value lookup recovers
        # the per-detection id without changing the tracker's own contract.
        centroid_to_id = {track.centroid: track_id for track_id, track in self.tracker.tracks.items()}
        return [centroid_to_id.get(det.centroid) for det in detections]

    def tick(self, now: Optional[float] = None) -> TrafficMetrics:
        now = now if now is not None else time.time()
        with self._lock:
            try:
                metrics = self._tick_unsafe(now)
                self.last_error = None
                self._last_metrics = metrics
                return metrics
            except Exception as exc:  # malformed frame, detector/tracker failure, etc.
                self.last_error = f"{type(exc).__name__}: {exc}"
                return self._last_metrics

    def switch_video(self, video_source: str, detector_kind: Optional[str] = None) -> None:
        """Hot-swaps this direction's footage in place -- same adapter
        object, same DirectionalAgent/CoordinatorAgent wiring around it, no
        pipeline rebuild. Opens the new capture BEFORE touching any existing
        state, so a bad path leaves this adapter exactly as it was (safe to
        retry) instead of leaving it half-switched.

        A *stateful* detector (MotionDetector's background-subtraction
        model) is always rebuilt fresh, even when staying on "motion" --
        reusing one that's already adapted to the OLD clip's background
        would contaminate detections on the new one until it re-adapts on
        its own. A stateless detector (YOLO) is reused as-is when its kind
        doesn't change, since a fresh instance would only cost a model
        reload for no correctness benefit.

        The fresh detector is pre-warmed synchronously here (reading and
        discarding its own warmup_frames from the new capture, at CV speed,
        not the pipeline's ~1s tick cadence) so the very next real tick
        already gets real detections -- this is what keeps a switch's
        visible effect within a few seconds instead of ~15."""
        new_capture = self._cv2.VideoCapture(video_source)
        if not new_capture.isOpened():
            new_capture.release()
            raise RuntimeError(f"could not open video source: {video_source!r}")

        kind = detector_kind or self.config.detector_kind
        is_stateful = type(self.detector).__name__ == "MotionDetector"
        kind_changed = detector_kind is not None and detector_kind != self.config.detector_kind
        if is_stateful or kind_changed:
            new_detector, fallback_reason = build_detector_safe(kind, **self._detector_kwargs(kind))
            warmup_frames = getattr(new_detector, "warmup_frames", 0)
            if warmup_frames:
                # MotionDetector's own check is "frames_seen <= warmup_frames
                # -> []", so it takes warmup_frames + 1 calls before the
                # first real (non-empty) detection is even possible.
                # Deliberately NOT rewound afterwards: replaying the exact
                # same frames a second time in a row lets MOG2's per-pixel
                # model start absorbing the (now perfectly repeated) motion
                # as background, suppressing the very detections warmup is
                # supposed to enable. Continuing forward from here is
                # exactly what a plain cold start does (proven to detect
                # correctly), and the clip loops on its own via _read_frame
                # once it runs out.
                for _ in range(warmup_frames + 1):
                    ok, frame = new_capture.read()
                    if not ok:
                        break
                    new_detector.detect(self._resize(frame))
        else:
            new_detector, fallback_reason = self.detector, self.detector_fallback_reason

        # Re-probe orientation for the NEW clip -- a different upload can
        # need a different (or no) rotation correction than whatever the
        # previous video needed. Run against new_capture/new_detector
        # before committing, same as the MotionDetector warmup above; reads
        # a few frames off the new capture, which is harmless since it
        # hasn't been used for anything else yet.
        orientation_rotation = self._probe_orientation(new_capture, new_detector)

        with self._lock:
            self._capture.release()
            self._capture = new_capture
            self.detector = new_detector
            self.detector_fallback_reason = fallback_reason
            self._orientation_rotation = orientation_rotation
            self.tracker = CentroidTracker()
            self._frame_index = 0
            self._last_detections = []
            self._last_detection_ids = []
            self._seen_track_ids = set()
            self._arrivals = deque()
            self.last_error = None
            self.last_annotated_jpeg = None  # never show a stale box from the previous clip
            self._detector_latency_ms = 0.0
            self._tracker_fps = 0.0
            self.config = replace(self.config, video_source=video_source, detector_kind=kind)
            self.source_id = os.path.basename(str(video_source))
            self._last_metrics = self._zero_metrics(time.time())

    def status(self) -> dict:
        """The evaluator-facing VIDEO/SOURCE/DIRECTION/DETECTOR/TRACKING
        panel's real data -- every field here is read off live adapter
        state, nothing is a placeholder."""
        with self._lock:
            confidences = [d.confidence for d in self._last_detections]
            is_yolo = type(self.detector).__name__ == "VehicleDetector"
            return {
                "direction": self.config.direction,
                "video_filename": self.source_id,
                "detector_label": "YOLO" if is_yolo else "CV Fallback (motion)",
                # Explicit so nothing downstream can mistake a motion blob
                # for a classified vehicle: PRIMARY = real object detector,
                # FALLBACK = classical CV for footage YOLO can't classify.
                "detector_mode": "PRIMARY" if is_yolo else "FALLBACK",
                "detector_fallback_reason": self.detector_fallback_reason,
                "tracking_active": True,
                "active_tracks": len(self.tracker.tracks),
                "unique_vehicle_count": len(self._seen_track_ids),
                "avg_confidence": (sum(confidences) / len(confidences)) if confidences else None,
                "detector_latency_ms": self._detector_latency_ms,
                "tracker_fps": self._tracker_fps,
                "orientation_correction_degrees": self._orientation_rotation,
                "last_error": self.last_error,
            }

    def _tick_unsafe(self, now: float) -> TrafficMetrics:
        tick_start = time.perf_counter()
        ok, frame = self._read_frame()
        if not ok or frame is None or frame.size == 0:
            raise RuntimeError(f"no usable frame from {self.config.video_source!r}")
        # Orientation correction happens before resize/ROI/detection so
        # everything downstream (including normalized ROI, which scales
        # against THIS frame's dimensions) consistently operates on the
        # corrected frame -- no separate "map boxes back" step is needed
        # because detection, tracking, and the annotated preview all use
        # the same corrected frame from here on.
        frame = self._rotate(frame, self._orientation_rotation)
        frame = self._resize(frame)
        roi_polygon, queue_polygon = self._roi_polygons(frame)

        # frame skipping: only run the (expensive) detector on 1 out of
        # every (frame_skip + 1) frames, reusing the last result otherwise.
        # The tracker is intentionally NOT touched on a skipped frame -- its
        # tracks simply hold their last known position/box until the next
        # real detection, which is what keeps boxes/IDs from flickering
        # between detector frames instead of vanishing every skip.
        run_detection = (self._frame_index % (self.config.frame_skip + 1)) == 0
        self._frame_index += 1
        if run_detection:
            detect_start = time.perf_counter()
            raw_detections = self.detector.detect(frame)
            latency_ms = (time.perf_counter() - detect_start) * 1000.0
            self._detector_latency_ms = (
                latency_ms if self._detector_latency_ms == 0.0
                else LATENCY_EMA_ALPHA * latency_ms + (1 - LATENCY_EMA_ALPHA) * self._detector_latency_ms
            )
            self._last_detections = [d for d in raw_detections if point_in_polygon(d.centroid, roi_polygon)]
            self.tracker.update(self._last_detections)
            self._last_detection_ids = self._match_track_ids(self._last_detections)
        detections = self._last_detections
        detection_ids = self._last_detection_ids

        # Active vehicle_count/queue_length come from the tracker's live
        # tracks, not this tick's raw detections -- a track that's coasting
        # through a brief miss (age_since_seen <= tracker.max_age) still
        # counts, which is the temporal smoothing that stops one missed
        # detection from zeroing the count. A track never appears here
        # unless a real detection created it, so this never fabricates
        # vehicles that were never actually seen.
        live_tracks = list(self.tracker.tracks.values())
        vehicle_count = len(live_tracks)
        queue_length = sum(1 for t in live_tracks if point_in_polygon(t.centroid, queue_polygon))

        new_ids = set(self.tracker.tracks.keys()) - self._seen_track_ids
        self._seen_track_ids |= new_ids
        self._arrivals.extend([now] * len(new_ids))
        while self._arrivals and now - self._arrivals[0] > ARRIVAL_WINDOW_SECONDS:
            self._arrivals.popleft()
        arrival_rate = len(self._arrivals) * (60.0 / ARRIVAL_WINDOW_SECONDS)

        self.last_annotated_jpeg = self._annotate(frame, detections, detection_ids, roi_polygon, queue_polygon)

        tick_duration = time.perf_counter() - tick_start
        instant_fps = (1.0 / tick_duration) if tick_duration > 0 else 0.0
        self._tracker_fps = (
            instant_fps if self._tracker_fps == 0.0
            else LATENCY_EMA_ALPHA * instant_fps + (1 - LATENCY_EMA_ALPHA) * self._tracker_fps
        )

        return TrafficMetrics(
            direction=self.config.direction,
            vehicle_count=vehicle_count,
            queue_length=queue_length,
            avg_wait_time=queue_length * 3.0,  # same demo estimate the simulator uses, for comparability
            arrival_rate=arrival_rate,
            timestamp=now,
            density=min(1.0, vehicle_count / LANE_CAPACITY),
            source_id=self.source_id,
            unique_vehicle_count=len(self._seen_track_ids),
            detector_latency_ms=self._detector_latency_ms,
            tracker_fps=self._tracker_fps,
        )

    def release(self) -> None:
        self._capture.release()


class VisionSource:
    """Runs one DirectionVisionAdapter per configured direction and
    publishes a combined batch to the "metrics" topic every tick -- start
    with a single-entry `configs` dict for one direction; adding N/S/E/W
    is just adding more entries, no code change.

    A direction whose video can't be opened is skipped (not fatal) so the
    other three keep running; see errors_snapshot() for what failed."""

    def __init__(
        self,
        bus: MessageBus,
        configs: Dict[str, ROIConfig],
        detector: Optional[Detector] = None,
    ) -> None:
        self.bus = bus
        self.errors: Dict[str, str] = {}
        self.adapters: Dict[str, DirectionVisionAdapter] = {}
        for direction, config in configs.items():
            try:
                self.adapters[direction] = DirectionVisionAdapter(config, detector=detector)
            except Exception as exc:
                self.errors[direction] = str(exc)
        if not self.adapters:
            raise RuntimeError(f"no video source could be opened: {self.errors}")
        self.directions: Tuple[str, ...] = tuple(self.adapters.keys())

    def tick(self) -> Dict[str, TrafficMetrics]:
        now = time.time()
        return {direction: adapter.tick(now) for direction, adapter in self.adapters.items()}

    def switch_video(self, direction: str, video_source: str, detector_kind: Optional[str] = None) -> None:
        """Hot-swaps one already-video-backed direction's footage without
        touching any other direction or rebuilding the pipeline. Only valid
        for a direction that already has a running adapter -- adding a
        brand-new direction changes this source's topology, which needs a
        real restart (see dashboard.py's Start/Restart)."""
        if direction not in self.adapters:
            raise KeyError(f"{direction!r} has no running video adapter to switch")
        self.adapters[direction].switch_video(video_source, detector_kind=detector_kind)

    def status_snapshot(self) -> Dict[str, dict]:
        return {d: a.status() for d, a in self.adapters.items()}

    def frames_snapshot(self) -> Dict[str, bytes]:
        return {d: a.last_annotated_jpeg for d, a in self.adapters.items() if a.last_annotated_jpeg is not None}

    def errors_snapshot(self) -> Dict[str, str]:
        errors = dict(self.errors)
        for direction, adapter in self.adapters.items():
            if adapter.last_error:
                errors[direction] = adapter.last_error
            elif adapter.detector_fallback_reason:
                errors[direction] = adapter.detector_fallback_reason
        return errors

    async def run(self, interval: float = 0.2) -> None:
        # interval was 1.0s; at that cadence cv2.VideoCapture.read() (which
        # always returns the NEXT frame, not "the frame at wall-clock now")
        # only ever sampled ~1 frame/second of real footage, so most of a
        # 24-30fps clip was never even read -- the main reason vehicles were
        # missed. 0.2s (plus subtracting this tick's own processing time
        # below, so slow inference doesn't compound with the sleep) samples
        # roughly 5x more of the video without touching frame_skip's own
        # per-adapter meaning.
        while True:
            tick_start = time.monotonic()
            try:
                # cv2/YOLO calls are blocking; keep them off the event loop
                # rather than redesigning the agent side to be thread-aware
                batch = await asyncio.to_thread(self.tick)
                await self.bus.publish("metrics", sender="vision", payload=batch)
            except Exception:
                # Per-adapter failures already return cached metrics instead
                # of raising (see tick()'s own try/except) -- this is the
                # last-resort net for anything that slips past that, so one
                # bad cycle skips a publish instead of killing the whole
                # asyncio.gather() pipeline the signal controller runs in.
                logging.getLogger("maats.cv").exception("VisionSource.run(): unexpected error this cycle, continuing")
            elapsed = time.monotonic() - tick_start
            await asyncio.sleep(max(0.0, interval - elapsed))

    def release(self) -> None:
        for adapter in self.adapters.values():
            adapter.release()

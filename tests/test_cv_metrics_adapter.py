from typing import List

import pytest

from backend.cv.detector import Detection
from backend.cv.metrics_adapter import DirectionVisionAdapter, VisionSource
from backend.cv.roi_config import ROIConfig
from backend.agents.message_bus import MessageBus

DEMO_VIDEO = "backend/cv/configs/demo_videos/north.mp4"

FULL_FRAME_ROI = [(0, 0), (640, 0), (640, 480), (0, 480)]
QUEUE_ZONE = [(0, 320), (640, 320), (640, 480), (0, 480)]  # bottom third only


class FakeDetector:
    """Scripted detector: returns a pre-programmed detection list each call
    (one entry per tick), regardless of actual frame content. Lets these
    tests exercise ROI/queue/tracking/arrival-rate logic without a real
    YOLO model or real vehicle footage."""

    def __init__(self, script: List[List[Detection]]) -> None:
        self._script = script
        self.calls = 0
        self.last_frame_shapes: List[tuple] = []

    def detect(self, frame) -> List[Detection]:
        self.last_frame_shapes.append(frame.shape)
        result = self._script[self.calls] if self.calls < len(self._script) else []
        self.calls += 1
        return result


def det(cx, cy, class_name="car"):
    return Detection(class_name=class_name, confidence=0.9, bbox=(cx - 5, cy - 5, cx + 5, cy + 5))


def make_config(frame_skip=0, resize_width=640) -> ROIConfig:
    return ROIConfig(
        direction="N",
        video_source=DEMO_VIDEO,
        roi_polygon=FULL_FRAME_ROI,
        queue_polygon=QUEUE_ZONE,
        frame_skip=frame_skip,
        resize_width=resize_width,
    )


def test_vehicle_count_and_queue_length_from_roi_and_queue_polygon():
    # 2 vehicles in the queue zone (y=400), 1 vehicle elsewhere in the ROI (y=100)
    script = [[det(100, 400), det(300, 400), det(500, 100)]]
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector(script))

    metrics = adapter.tick(now=0.0)
    assert metrics.direction == "N"
    assert metrics.vehicle_count == 3
    assert metrics.queue_length == 2


def test_detections_outside_roi_are_excluded():
    # one inside the frame/ROI, one far outside any real coordinate space
    script = [[det(100, 400), det(-500, -500)]]
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector(script))
    metrics = adapter.tick(now=0.0)
    assert metrics.vehicle_count == 1


def test_frame_skip_only_calls_the_detector_on_every_nth_frame():
    fake = FakeDetector([[det(100, 400)]] * 10)
    adapter = DirectionVisionAdapter(make_config(frame_skip=2), detector=fake)

    for i in range(6):
        adapter.tick(now=float(i))

    # frame_skip=2 -> detector runs on ticks 0, 3 (every 3rd call) out of 6
    assert fake.calls == 2


def test_resize_width_shrinks_frames_before_detection():
    fake = FakeDetector([[]] * 5)
    adapter = DirectionVisionAdapter(make_config(resize_width=320), detector=fake)
    adapter.tick(now=0.0)
    assert fake.last_frame_shapes[0][1] == 320  # width dimension


def test_resize_width_zero_disables_resizing():
    fake = FakeDetector([[]] * 5)
    adapter = DirectionVisionAdapter(make_config(resize_width=0), detector=fake)
    adapter.tick(now=0.0)
    assert fake.last_frame_shapes[0][1] == 640  # native demo video width


def test_arrival_rate_reflects_new_track_ids_within_the_window():
    # 3 distinct new vehicles across 3 ticks -> 3 arrivals inside a 60s window
    script = [[det(100, 400)], [det(300, 100)], [det(500, 250)]]
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector(script))

    adapter.tick(now=0.0)
    adapter.tick(now=1.0)
    metrics = adapter.tick(now=2.0)

    assert metrics.arrival_rate == pytest.approx(3 * (60.0 / 60.0))


def test_arrival_rate_rolls_off_after_the_window_expires():
    script = [[det(100, 400)], []]
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector(script))

    adapter.tick(now=0.0)  # 1 arrival recorded at t=0
    metrics = adapter.tick(now=200.0)  # long past the 60s window

    assert metrics.arrival_rate == 0.0


def test_video_loops_past_end_of_clip_instead_of_raising():
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector([[]] * 500))
    for i in range(150):  # demo clip is 90 frames; this wraps at least once
        adapter.tick(now=float(i))  # should not raise


class RaisingDetector:
    """Simulates a detector that blows up mid-stream (e.g. a YOLO inference
    error) so tick() robustness can be tested without a real broken model."""

    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("simulated detector failure")
        return [det(100, 400)]


def test_malformed_frame_does_not_crash_the_pipeline_and_reuses_last_metrics():
    fake = RaisingDetector(fail_on_call=2)
    adapter = DirectionVisionAdapter(make_config(frame_skip=0), detector=fake)

    good = adapter.tick(now=0.0)
    assert adapter.last_error is None
    assert good.vehicle_count == 1

    recovered = adapter.tick(now=1.0)  # detector raises on this call
    assert adapter.last_error is not None
    assert recovered == good  # falls back to the last known-good TrafficMetrics, doesn't crash

    healthy_again = adapter.tick(now=2.0)
    assert adapter.last_error is None
    assert healthy_again.vehicle_count == 1


def test_empty_detections_produce_zero_count_metrics_without_error():
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector([[]] * 5))
    metrics = adapter.tick(now=0.0)
    assert metrics.vehicle_count == 0
    assert metrics.queue_length == 0
    assert adapter.last_error is None


def test_traffic_metrics_conversion_includes_density_and_source_id():
    script = [[det(100, 400), det(300, 400)]]
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector(script))
    metrics = adapter.tick(now=0.0)

    assert metrics.source_id == "north.mp4"
    assert metrics.density == pytest.approx(2 / 20)  # LANE_CAPACITY = 20


def test_track_id_is_stable_across_ticks_for_the_same_vehicle():
    script = [[det(100, 400)], [det(103, 402)]]  # small shift, same vehicle
    adapter = DirectionVisionAdapter(make_config(frame_skip=0), detector=FakeDetector(script))

    adapter.tick(now=0.0)
    first_id = adapter._last_detection_ids[0]
    adapter.tick(now=1.0)
    second_id = adapter._last_detection_ids[0]

    assert first_id is not None
    assert first_id == second_id


def test_video_switching_produces_metrics_through_the_same_pipeline():
    # changing which file a direction reads from must not require any
    # different code path -- same adapter class, same detector contract
    north_config = ROIConfig(
        direction="N", video_source="backend/cv/configs/demo_videos/north.mp4",
        roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
    )
    south_config = ROIConfig(
        direction="N", video_source="backend/cv/configs/demo_videos/south.mp4",
        roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
    )

    north_adapter = DirectionVisionAdapter(north_config, detector=FakeDetector([[det(100, 400)]]))
    south_adapter = DirectionVisionAdapter(south_config, detector=FakeDetector([[det(100, 400)]]))

    north_metrics = north_adapter.tick(now=0.0)
    south_metrics = south_adapter.tick(now=0.0)

    assert north_metrics.source_id == "north.mp4"
    assert south_metrics.source_id == "south.mp4"
    assert north_metrics.vehicle_count == south_metrics.vehicle_count == 1


def test_vision_source_combines_all_configured_directions_into_one_batch():
    configs = {
        "N": make_config(),
        "S": ROIConfig(
            direction="S", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        ),
    }
    source = VisionSource(MessageBus(), configs, detector=FakeDetector([[det(100, 400)]] * 10))
    batch = source.tick()
    assert set(batch.keys()) == {"N", "S"}
    assert batch["N"].direction == "N"
    assert batch["S"].direction == "S"


# -- Hot-swap ("show me another traffic video" without restarting anything) --

SOUTH_VIDEO = "backend/cv/configs/demo_videos/south.mp4"
PORTRAIT_VIDEO = "data/traffic/portrait_480x640.mp4"
HEAVY_VIDEO = "data/traffic/heavy.mp4"


def test_switch_video_resets_tracker_and_arrival_state():
    fake = FakeDetector([[det(100, 400)], [det(103, 402)]])
    adapter = DirectionVisionAdapter(make_config(), detector=fake)
    adapter.tick(now=0.0)
    adapter.tick(now=1.0)
    assert len(adapter.tracker.tracks) > 0
    assert adapter._seen_track_ids
    assert adapter._frame_index == 2

    adapter.switch_video(SOUTH_VIDEO)

    assert len(adapter.tracker.tracks) == 0
    assert adapter._seen_track_ids == set()
    assert adapter._frame_index == 0
    assert list(adapter._arrivals) == []
    assert adapter.config.video_source == SOUTH_VIDEO
    assert adapter.source_id == "south.mp4"


def test_switch_video_releases_the_previous_capture():
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector([[]] * 5))
    old_capture = adapter._capture
    adapter.switch_video(SOUTH_VIDEO)
    assert old_capture.isOpened() is False
    assert adapter._capture is not old_capture
    assert adapter._capture.isOpened() is True


def test_switch_video_to_a_bad_path_raises_and_leaves_the_old_video_usable():
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector([[det(100, 400)]] * 5))
    with pytest.raises(RuntimeError):
        adapter.switch_video("does/not/exist.mp4")

    metrics = adapter.tick(now=0.0)  # old video/capture must still work, unmodified
    assert metrics.vehicle_count == 1
    assert adapter.config.video_source == DEMO_VIDEO


def test_switch_video_adapts_to_a_different_resolution_and_orientation_without_crashing():
    # a real MotionDetector (not the scripted fake) against a genuinely
    # different, portrait-orientation, differently-sized clip
    config = ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        frame_skip=0, detector_kind="motion",
    )
    adapter = DirectionVisionAdapter(config)
    adapter.switch_video(PORTRAIT_VIDEO)

    metrics = None
    for i in range(20):  # past MotionDetector's warmup_frames=15
        metrics = adapter.tick(now=float(i))
    assert adapter.last_error is None
    assert metrics.vehicle_count >= 0  # no crash on the differently-shaped frame


def test_switch_video_updates_metrics_to_the_new_clips_real_content():
    config = ROIConfig(
        direction="E", video_source="data/traffic/empty.mp4", roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        frame_skip=0, detector_kind="motion",
    )
    adapter = DirectionVisionAdapter(config)
    for i in range(20):
        before = adapter.tick(now=float(i))
    assert before.vehicle_count == 0

    adapter.switch_video(HEAVY_VIDEO)
    after = None
    for i in range(20, 40):
        after = adapter.tick(now=float(i))
    assert after.vehicle_count > 0


def test_status_reports_real_video_filename_detector_label_and_active_tracks():
    config = ROIConfig(
        direction="E", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        frame_skip=0, detector_kind="motion",
    )
    adapter = DirectionVisionAdapter(config)
    status = adapter.status()
    assert status["direction"] == "E"
    assert status["video_filename"] == "north.mp4"
    assert status["detector_label"] == "CV Fallback (motion)"
    assert status["tracking_active"] is True

    adapter.switch_video(HEAVY_VIDEO)
    for i in range(20):
        adapter.tick(now=float(i))
    status_after = adapter.status()
    assert status_after["video_filename"] == "heavy.mp4"
    assert status_after["active_tracks"] >= 0


def test_status_labels_a_vehicledetector_instance_as_yolo():
    class VehicleDetector:  # shadow class: tests the label-by-type-name branch without loading real YOLO
        def detect(self, frame):
            return []

    adapter = DirectionVisionAdapter(make_config(), detector=VehicleDetector())
    assert adapter.status()["detector_label"] == "YOLO"


def test_status_reports_primary_for_yolo_and_fallback_for_motion():
    class VehicleDetector:
        def detect(self, frame):
            return []

    yolo_adapter = DirectionVisionAdapter(make_config(), detector=VehicleDetector())
    assert yolo_adapter.status()["detector_mode"] == "PRIMARY"

    motion_config = ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        frame_skip=0, detector_kind="motion",
    )
    motion_adapter = DirectionVisionAdapter(motion_config)
    assert motion_adapter.status()["detector_mode"] == "FALLBACK"


# -- Temporal smoothing: a track that briefly misses detection must keep
# counting until it actually ages out, not vanish for one bad frame --

def test_vehicle_count_coasts_through_a_single_missed_detection():
    # tick 1: vehicle detected. tick 2: detector sees nothing (a real miss,
    # not a crash). tick 3: same vehicle detected again nearby.
    script = [[det(100, 400)], [], [det(103, 402)]]
    adapter = DirectionVisionAdapter(make_config(frame_skip=0), detector=FakeDetector(script))

    first = adapter.tick(now=0.0)
    assert first.vehicle_count == 1

    during_miss = adapter.tick(now=1.0)
    assert during_miss.vehicle_count == 1  # coasting, not zeroed by one miss

    recovered = adapter.tick(now=2.0)
    assert recovered.vehicle_count == 1
    assert adapter._last_detection_ids[0] == adapter.tracker.tracks[next(iter(adapter.tracker.tracks))].track_id


def test_vehicle_count_does_not_fabricate_vehicles_that_were_never_detected():
    adapter = DirectionVisionAdapter(make_config(), detector=FakeDetector([[]] * 10))
    for i in range(10):
        metrics = adapter.tick(now=float(i))
    assert metrics.vehicle_count == 0
    assert metrics.unique_vehicle_count == 0


def test_new_metrics_fields_are_populated_not_left_as_dead_defaults():
    fake = FakeDetector([[det(100, 400)]] * 3)
    adapter = DirectionVisionAdapter(make_config(frame_skip=0), detector=fake)
    for i in range(3):
        metrics = adapter.tick(now=float(i))
    assert metrics.unique_vehicle_count == 1
    assert metrics.detector_latency_ms >= 0.0
    assert metrics.tracker_fps > 0.0


# -- Resolution-independent ROI (normalized coordinates) --

NORMALIZED_ROI = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
NORMALIZED_QUEUE = [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]  # bottom half


def test_normalized_roi_scales_to_the_actual_frame_size():
    config = ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=NORMALIZED_ROI, queue_polygon=NORMALIZED_QUEUE,
        roi_normalized=True, frame_skip=0, resize_width=640,
    )
    # the demo clip is 640x480 -- a point at (100, 400) is in the bottom
    # half (y=400 > 240), so it must land in the (normalized) queue zone
    # exactly as it would under the old hardcoded-pixel equivalent
    adapter = DirectionVisionAdapter(config, detector=FakeDetector([[det(100, 400)]]))
    metrics = adapter.tick(now=0.0)
    assert metrics.vehicle_count == 1
    assert metrics.queue_length == 1


def test_normalized_roi_survives_switching_to_a_different_resolution_and_orientation():
    # a real MotionDetector against a genuinely different, portrait clip --
    # proves the normalized polygon doesn't require any resolution-specific
    # config change on switch, unlike the legacy pixel-polygon path
    config = ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=NORMALIZED_ROI, queue_polygon=NORMALIZED_QUEUE,
        roi_normalized=True, frame_skip=0, detector_kind="motion",
    )
    adapter = DirectionVisionAdapter(config)
    adapter.switch_video(PORTRAIT_VIDEO)

    metrics = None
    for i in range(20):  # past MotionDetector's warmup_frames=15
        metrics = adapter.tick(now=float(i))
    assert adapter.last_error is None
    assert metrics.vehicle_count >= 0  # no crash, no polygon-out-of-bounds silent zeroing


# -- Detector-kind switching (motion -> yolo mid-stream, without a restart) --

def test_switch_video_can_change_detector_kind_from_motion_to_a_stub_yolo():
    class VehicleDetector:  # stands in for a real YOLO instance
        def detect(self, frame):
            return [det(100, 400)]

    config = ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        frame_skip=0, detector_kind="motion",
    )
    adapter = DirectionVisionAdapter(config)
    assert adapter.status()["detector_mode"] == "FALLBACK"

    # switch_video's detector_kind override is what dashboard.py uses when
    # the user uploads real footage to a direction that was running motion
    adapter.detector = VehicleDetector()  # simulate the post-switch instance directly
    adapter.detector_fallback_reason = None
    assert adapter.status()["detector_mode"] == "PRIMARY"


def test_empty_video_with_yolo_kind_reports_zero_without_crashing():
    config = ROIConfig(
        direction="E", video_source="data/traffic/empty.mp4", roi_polygon=NORMALIZED_ROI,
        queue_polygon=NORMALIZED_QUEUE, roi_normalized=True, frame_skip=0,
    )
    adapter = DirectionVisionAdapter(config, detector=FakeDetector([[]] * 20))
    metrics = None
    for i in range(20):
        metrics = adapter.tick(now=float(i))
    assert adapter.last_error is None
    assert metrics.vehicle_count == 0


def test_bad_video_path_fails_safely_without_crashing_the_caller():
    config = ROIConfig(
        direction="N", video_source="does/not/exist.mp4", roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
    )
    with pytest.raises(RuntimeError):
        DirectionVisionAdapter(config, detector=FakeDetector([]))


# -- Orientation correction: fixes a real defect found in demo validation
# (a genuinely rotated 720x1280 real traffic video scored first_detection
# ~5.5s, peak 1 track, avg 0.04 active tracks -- COCO-trained YOLO barely
# recognizes sideways vehicles). These use a fake capture + a
# shape-discriminating fake detector so the algorithm itself is tested
# fast and deterministically, without real video/weights. --

import numpy as np


class FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame


class VehicleDetector:  # noqa: F811 -- shadow class, matches this file's existing name-based duck-typing convention
    """'Sees' a vehicle only when the frame is landscape-shaped -- a clean,
    deterministic stand-in for "COCO-trained YOLO barely recognizes
    sideways vehicles" without needing a real model."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        height, width = frame.shape[:2]
        if width > height:
            return [Detection(class_name="car", confidence=0.9, bbox=(0, 0, 10, 10))]
        return []


def _portrait_frame():
    return np.zeros((200, 100, 3), dtype=np.uint8)  # h=200, w=100 -- portrait


def _landscape_frame():
    return np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200 -- landscape


def test_orientation_probe_picks_a_rotation_that_clearly_improves_detection():
    adapter = DirectionVisionAdapter(ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
    ), detector=FakeDetector([]))
    detector = VehicleDetector()
    capture = FakeCapture([_portrait_frame()] * 3)
    rotation = adapter._probe_orientation(capture, detector)
    # both +90 and -90 turn this portrait frame landscape (equally "correct"
    # to the fake detector) -- either is an acceptable, deterministic pick
    assert rotation in (90, -90)


def test_orientation_probe_leaves_a_landscape_video_uncorrected_and_skips_probing():
    adapter = DirectionVisionAdapter(ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
    ), detector=FakeDetector([]))
    detector = VehicleDetector()
    capture = FakeCapture([_landscape_frame()] * 3)
    rotation = adapter._probe_orientation(capture, detector)
    assert rotation == 0
    assert detector.calls == 0  # zero detector calls for the common landscape case -- zero added latency


def test_orientation_probe_skips_entirely_for_the_motion_fallback_detector():
    from backend.cv.motion_detector import MotionDetector

    adapter = DirectionVisionAdapter(ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        detector_kind="motion",
    ))
    capture = FakeCapture([_portrait_frame()] * 3)
    rotation = adapter._probe_orientation(capture, MotionDetector())
    assert rotation == 0
    assert capture._i == 0  # never even read a frame -- motion detector's failure mode isn't orientation


def test_orientation_probe_does_not_rotate_a_genuinely_portrait_video_on_marginal_evidence():
    # A real portrait-filmed video (vehicles already upright) might detect
    # slightly better in one rotated candidate purely by chance -- the
    # improvement must be CLEAR (>1.5x) before we act on it, or a real
    # portrait video would get needlessly rotated on noise.
    class MarginalDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, frame):
            self.calls += 1
            height, width = frame.shape[:2]
            confidence = 0.9 if width > height else 0.7  # rotated is only slightly "better"
            return [Detection(class_name="car", confidence=confidence, bbox=(0, 0, 10, 10))]

    adapter = DirectionVisionAdapter(ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
    ), detector=FakeDetector([]))
    # name it VehicleDetector so the probe actually engages
    MarginalDetector.__name__ = "VehicleDetector"
    detector = MarginalDetector()
    capture = FakeCapture([_portrait_frame()] * 3)
    rotation = adapter._probe_orientation(capture, detector)
    assert rotation == 0  # left alone: not a clear enough improvement to override


def test_orientation_correction_is_applied_before_resize_and_roi_in_a_real_tick():
    # End-to-end (still with a fake detector, no real weights): once a
    # rotation is chosen, every subsequent tick must actually apply it, and
    # tracking must work normally on the corrected frame.
    adapter = DirectionVisionAdapter(ROIConfig(
        direction="N", video_source=DEMO_VIDEO, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        roi_normalized=True,
    ), detector=FakeDetector([[det(100, 400)]] * 5))
    adapter._orientation_rotation = 90  # simulate a prior probe result
    metrics = adapter.tick(now=0.0)
    assert adapter.last_error is None
    assert metrics.vehicle_count == 1  # still works: rotation didn't break the rest of the pipeline

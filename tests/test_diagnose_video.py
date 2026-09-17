"""Coverage for scripts/diagnose_video.py: the report shape the CLI prints,
safe failure on a corrupted/missing video, and a loose performance-sanity
bound (catches a gross regression, e.g. someone reintroducing a per-frame
model reload, without being flaky about exact hardware speed).
"""
from __future__ import annotations

import time

import pytest

from scripts.diagnose_video import diagnose

DEMO_VIDEO = "backend/cv/configs/demo_videos/north.mp4"  # 640x480, 90 frames, motion-detector clip


def test_diagnose_reports_the_expected_fields_on_a_motion_clip():
    report = diagnose(DEMO_VIDEO, detector_kind="motion", frame_skip=0, max_frames=30)
    for key in (
        "frames_processed", "total_raw_detections", "unique_tracks",
        "avg_detector_latency_ms", "avg_processing_fps",
        "min_confidence", "max_confidence", "vehicle_counts_over_time_sampled",
    ):
        assert key in report
    assert report["frames_processed"] == 30


def test_diagnose_fails_safely_on_a_nonexistent_video():
    with pytest.raises(SystemExit):
        diagnose("does/not/exist.mp4")


def test_diagnose_does_not_fabricate_detections_on_an_empty_clip():
    report = diagnose("data/traffic/empty.mp4", detector_kind="motion", frame_skip=0, max_frames=30)
    assert report["peak_active_vehicle_count"] == 0
    assert report["min_confidence"] is None  # no detections at all -- not a fabricated 0.0


def test_processing_a_short_clip_completes_within_a_generous_time_bound():
    # performance sanity: 30 frames of a real YOLO pass on a 640x480 clip
    # should not take anywhere near this long -- a regression like reloading
    # the model every frame would blow well past it
    start = time.perf_counter()
    diagnose(
        "backend/cv/configs/uploads/N_2165-155327596_medium.mp4",
        weights="yolov8n.pt", frame_skip=0, max_frames=30,
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0

"""VehicleDetector-specific coverage: initialization, class filtering,
confidence filtering, and the weights-cache that makes "load once, reuse"
true across multiple directions. Uses the real yolov8n.pt already checked
into the repo root and real footage (no mocks) -- these are the properties
that can only be proven against the actual ultralytics model.
"""
from __future__ import annotations

import cv2

from backend.cv.detector import VEHICLE_CLASSES, VehicleDetector, _MODEL_CACHE, _load_yolo_cached

REAL_VIDEO = "backend/cv/configs/uploads/N_2165-155327596_medium.mp4"
WEIGHTS = "yolov8n.pt"


def _first_frame(path: str):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"could not read a frame from {path!r}"
    return frame


def test_detector_initializes_with_configurable_thresholds():
    detector = VehicleDetector(weights=WEIGHTS, confidence_threshold=0.4, iou_threshold=0.5)
    assert detector.confidence_threshold == 0.4
    assert detector.iou_threshold == 0.5


def test_supported_classes_are_the_four_vehicle_types_by_default():
    detector = VehicleDetector(weights=WEIGHTS)
    assert detector.vehicle_classes == VEHICLE_CLASSES
    assert detector.vehicle_classes == {"car", "motorcycle", "bus", "truck"}
    assert "bicycle" not in detector.vehicle_classes
    assert "person" not in detector.vehicle_classes  # unrelated COCO classes must be ignored


def test_include_bicycle_widens_the_class_set_when_requested():
    detector = VehicleDetector(weights=WEIGHTS, include_bicycle=True)
    assert "bicycle" in detector.vehicle_classes


def test_detect_only_returns_vehicle_classes_on_real_footage():
    detector = VehicleDetector(weights=WEIGHTS)
    frame = _first_frame(REAL_VIDEO)
    detections = detector.detect(frame)
    for det in detections:
        assert det.class_name in VEHICLE_CLASSES  # e.g. never "person" or "traffic light"


def test_confidence_filtering_respects_the_configured_threshold():
    frame = _first_frame(REAL_VIDEO)
    loose = VehicleDetector(weights=WEIGHTS, confidence_threshold=0.1).detect(frame)
    strict = VehicleDetector(weights=WEIGHTS, confidence_threshold=0.8).detect(frame)

    assert all(d.confidence >= 0.1 for d in loose)
    assert all(d.confidence >= 0.8 for d in strict)
    # a stricter threshold can only ever keep the same or fewer detections
    assert len(strict) <= len(loose)


def test_model_weights_are_loaded_once_and_reused_across_instances():
    _MODEL_CACHE.pop(WEIGHTS, None)  # isolate from whatever earlier tests loaded
    first = _load_yolo_cached(WEIGHTS)
    second = _load_yolo_cached(WEIGHTS)
    assert first is second  # same object, not a second model load

    detector_a = VehicleDetector(weights=WEIGHTS)
    detector_b = VehicleDetector(weights=WEIGHTS)
    assert detector_a.model is detector_b.model  # two "directions" share one loaded model

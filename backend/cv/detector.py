"""Thin wrapper around Ultralytics YOLO for vehicle detection.

Only turns a frame into a list of vehicle detections -- no tracking, no
ROI logic, no TrafficMetrics. Ultralytics is imported lazily inside
VehicleDetector.__init__ so this module (and the Detection dataclass /
VEHICLE_CLASSES constant) stays importable and unit-testable even before
`pip install ultralytics` has been run.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
OPTIONAL_VEHICLE_CLASSES = {"bicycle"}  # only added when include_bicycle=True

# Detector weights must load once and be reused -- every VehicleDetector
# built for the same weights path (e.g. one per direction) shares the same
# underlying ultralytics YOLO object instead of loading the model N times.
# Safe because DirectionVisionAdapter/VisionSource/HybridSource construct
# adapters sequentially, and inference calls happen one at a time per
# pipeline tick (see metrics_adapter.py) -- never concurrently from two
# threads at once.
_MODEL_CACHE: Dict[str, object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_yolo_cached(weights: str):
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(weights)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(weights)
            _MODEL_CACHE[weights] = model
        return model


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    track_id: Optional[int] = None  # filled in by the tracker, not the detector

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@runtime_checkable
class Detector(Protocol):
    """Common contract every detector implementation satisfies -- YOLO,
    the classical-CV fallback, or a test double. Nothing downstream
    (tracker, metrics adapter, agents) is allowed to know or care which
    one it's talking to."""

    def detect(self, frame) -> List[Detection]: ...


class VehicleDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",  # benchmarked against yolov8s (scripts/diagnose_video.py): n keeps
        # ~2.2x lower latency for comparable-or-better detection counts on the real demo clips
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        include_bicycle: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.model = _load_yolo_cached(weights)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.vehicle_classes = VEHICLE_CLASSES | (OPTIONAL_VEHICLE_CLASSES if include_bicycle else set())
        # Resolve class names -> COCO ids once so every detect() call can
        # pass classes=[...] straight to ultralytics -- it skips NMS/output
        # work for every non-vehicle class instead of filtering after the
        # fact, and avoids re-deriving this mapping every frame.
        self._class_ids = [i for i, name in self.model.names.items() if name in self.vehicle_classes]

    def detect(self, frame) -> List[Detection]:
        predict_kwargs = dict(
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self._class_ids,
        )
        if self.device is not None:
            predict_kwargs["device"] = self.device
        results = self.model(frame, **predict_kwargs)
        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_name = names[int(box.cls[0])]
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(Detection(class_name=class_name, confidence=confidence, bbox=(x1, y1, x2, y2)))
        return detections

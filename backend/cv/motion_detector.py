"""Classical-CV fallback detector for footage where YOLO's COCO vehicle
classes don't apply -- e.g. schematic/synthetic clips that represent traffic
as plain colored boxes rather than photographic cars.

Background subtraction (MOG2) + morphology + contour filtering by area and
rectangularity. Same detect() interface as VehicleDetector (see detector.py's
Detector protocol), so nothing downstream cares which one produced a
Detection. No ML model, no extra dependency -- OpenCV is already required.

Confirmed against the actual MAATS demo clips (backend/cv/configs/demo_videos):
YOLOv8n finds 0 detections on them (solid rectangles on a near-black
background aren't in COCO's vehicle classes), while this detector reliably
finds both moving boxes once the background model warms up.
"""
from __future__ import annotations

from typing import List

from .detector import Detection


class MotionDetector:
    def __init__(
        self,
        min_area: float = 200.0,
        max_area: float = 40000.0,
        min_aspect: float = 0.2,
        max_aspect: float = 5.0,
        min_rectangularity: float = 0.5,
        var_threshold: float = 32.0,
        warmup_frames: int = 15,
    ) -> None:
        import cv2

        self._cv2 = cv2
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=var_threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_rectangularity = min_rectangularity
        self.warmup_frames = warmup_frames
        self._frames_seen = 0

    def detect(self, frame) -> List[Detection]:
        cv2 = self._cv2
        self._frames_seen += 1

        foreground = self._bg_subtractor.apply(frame)
        # background model hasn't converged yet -- every pixel looks like
        # foreground on frame 1, so skip detections until it settles rather
        # than reporting a spurious full-frame box
        if self._frames_seen <= self.warmup_frames:
            return []

        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self._kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self._kernel)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0:
                continue
            aspect = w / h
            if not (self.min_aspect <= aspect <= self.max_aspect):
                continue
            rectangularity = area / (w * h)
            if rectangularity < self.min_rectangularity:
                continue
            # rectangularity doubles as a confidence proxy: a clean box-like
            # blob scores near 1.0, a ragged/noisy one scores lower
            detections.append(
                Detection(
                    class_name="object",
                    confidence=min(1.0, rectangularity),
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                )
            )
        return detections

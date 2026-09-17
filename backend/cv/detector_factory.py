"""Builds a Detector by name, so ROIConfig can say which one a given camera
feed needs ("yolo" for real footage, "motion" for synthetic/box-style demo
clips) without VisionSource/DirectionVisionAdapter caring which concrete
class comes back.

build_detector_safe() additionally never lets a broken YOLO install (missing
weights, missing ultralytics, etc.) take down the whole pipeline -- it falls
back to the dependency-free MotionDetector and reports why, satisfying "must
not crash if YOLO fails".
"""
from __future__ import annotations

from typing import Optional, Tuple

from .detector import Detector

DETECTOR_KINDS: Tuple[str, ...] = ("yolo", "motion")


def build_detector(kind: str = "yolo", **kwargs) -> Detector:
    if kind == "yolo":
        from .detector import VehicleDetector

        return VehicleDetector(**kwargs)
    if kind == "motion":
        from .motion_detector import MotionDetector

        return MotionDetector(**kwargs)
    raise ValueError(f"unknown detector kind: {kind!r}; choose from {DETECTOR_KINDS}")


def build_detector_safe(kind: str = "yolo", **kwargs) -> Tuple[Detector, Optional[str]]:
    """Returns (detector, fallback_reason). fallback_reason is None unless
    the requested kind failed to construct and MotionDetector was used
    instead."""
    try:
        return build_detector(kind, **kwargs), None
    except Exception as exc:  # e.g. ultralytics missing, weights unreadable
        if kind == "motion":
            raise  # motion has no further fallback; a real bug, don't hide it
        from .motion_detector import MotionDetector

        return MotionDetector(), f"{kind} detector failed ({exc}); using motion fallback"

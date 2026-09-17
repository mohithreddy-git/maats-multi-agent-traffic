"""Diagnostic CLI: runs one video through the real MAATS CV pipeline
(DirectionVisionAdapter -- the exact detector -> confidence filter -> ROI
filter -> tracker -> TrafficMetrics code path production uses, not a
reimplementation) and reports what actually happened.

Also doubles as the model-selection benchmark tool: run it once per
candidate weights file against the same video and compare
avg_detector_latency_ms / avg_processing_fps / confidence spread to decide
nano vs small without guessing.

Usage:
    python scripts/diagnose_video.py <video> [options]
    python scripts/diagnose_video.py backend/cv/configs/uploads/N_2165-155327596_medium.mp4 --weights yolov8s.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.cv.metrics_adapter import DirectionVisionAdapter  # noqa: E402
from backend.cv.roi_config import ROIConfig  # noqa: E402

# Normalized full-frame ROI/queue-zone -- resolution/orientation independent,
# appropriate for "just tell me what the detector sees" on an arbitrary
# uploaded clip (see ROIConfig.roi_normalized).
FULL_ROI = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
QUEUE_ZONE = [(0.0, 0.66), (1.0, 0.66), (1.0, 1.0), (0.0, 1.0)]


def diagnose(
    video_path: str,
    weights: str = "yolov8n.pt",
    confidence: float = 0.25,  # matches ROIConfig's own default -- see roi_config.py's comment for why
    iou: float = 0.45,
    resize_width: int = 640,
    frame_skip: int = 0,
    detector_kind: str = "yolo",
    max_frames: Optional[int] = None,
    include_bicycle: bool = False,
) -> dict:
    import cv2

    probe = cv2.VideoCapture(video_path)
    if not probe.isOpened():
        probe.release()
        raise SystemExit(f"could not open video: {video_path!r} (corrupted or unsupported file)")
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    native_width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    config = ROIConfig(
        direction="DIAG",
        video_source=video_path,
        roi_polygon=FULL_ROI,
        queue_polygon=QUEUE_ZONE,
        roi_normalized=True,
        frame_skip=frame_skip,
        resize_width=resize_width,
        detector_kind=detector_kind,
        confidence_threshold=confidence,
        iou_threshold=iou,
    )

    if detector_kind == "yolo":
        from backend.cv.detector import VehicleDetector

        detector = VehicleDetector(
            weights=weights, confidence_threshold=confidence, iou_threshold=iou, include_bicycle=include_bicycle,
        )
        adapter = DirectionVisionAdapter(config, detector=detector)
    else:
        adapter = DirectionVisionAdapter(config)  # motion fallback, built from config

    frames_to_process = max_frames or total_frames or 300
    confidences: List[float] = []
    vehicle_counts_over_time: List[int] = []

    wall_start = time.perf_counter()
    for i in range(frames_to_process):
        metrics = adapter.tick(now=float(i))
        confidences.extend(d.confidence for d in adapter._last_detections)
        vehicle_counts_over_time.append(metrics.vehicle_count)
    wall_elapsed = time.perf_counter() - wall_start

    status = adapter.status()
    adapter.release()

    sample_stride = max(1, len(vehicle_counts_over_time) // 20)
    return {
        "video": video_path,
        "native_resolution": f"{native_width}x{native_height}",
        "detector_kind": detector_kind,
        "weights": weights if detector_kind == "yolo" else "n/a",
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "resize_width": resize_width,
        "frame_skip": frame_skip,
        "frames_processed": frames_to_process,
        "total_raw_detections": len(confidences),
        "unique_tracks": status["unique_vehicle_count"],
        "avg_detector_latency_ms": round(status["detector_latency_ms"], 2),
        "avg_processing_fps": round(frames_to_process / wall_elapsed, 2) if wall_elapsed > 0 else 0.0,
        "min_confidence": round(min(confidences), 3) if confidences else None,
        "max_confidence": round(max(confidences), 3) if confidences else None,
        "final_active_vehicle_count": vehicle_counts_over_time[-1] if vehicle_counts_over_time else 0,
        "peak_active_vehicle_count": max(vehicle_counts_over_time) if vehicle_counts_over_time else 0,
        "vehicle_counts_over_time_sampled": vehicle_counts_over_time[::sample_stride],
        "wall_seconds": round(wall_elapsed, 2),
        "last_error": status.get("last_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MAATS vehicle detection/tracking on one video")
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--frame-skip", type=int, default=0)
    parser.add_argument("--detector-kind", default="yolo", choices=["yolo", "motion"])
    parser.add_argument("--max-frames", type=int, default=None, help="cap frames processed (default: whole clip)")
    parser.add_argument("--include-bicycle", action="store_true")
    args = parser.parse_args()

    report = diagnose(
        args.video,
        weights=args.weights,
        confidence=args.confidence,
        iou=args.iou,
        resize_width=args.resize_width,
        frame_skip=args.frame_skip,
        detector_kind=args.detector_kind,
        max_frames=args.max_frames,
        include_bicycle=args.include_bicycle,
    )
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

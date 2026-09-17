"""Per-direction ROI configuration.

Deliberately has no OpenCV/Ultralytics import at module load time: the
config schema and the point-in-polygon geometry test are pure Python, so
they stay testable even in an environment where those heavier libraries
aren't installed yet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class ROIConfig:
    direction: str
    video_source: str  # file path, or a camera index given as a string
    roi_polygon: List[Point]  # the full detection region for this approach
    queue_polygon: List[Point]  # sub-region near the stop line, used for queue_length
    frame_skip: int = 2  # run detection on 1 out of every (frame_skip + 1) frames
    resize_width: int = 640  # downscale to this width before inference; 0 disables
    detector_kind: str = "yolo"  # "yolo" or "motion" (classical-CV fallback for non-photographic footage)
    # 0.25, not ultralytics' usual 0.3 -- benchmarked on the two real
    # uploaded traffic clips (scripts/diagnose_video.py) at 0.3/0.25/0.2/0.15:
    # 0.25 recovers ~11% more real detections than 0.3 at zero latency cost
    # (confidence filtering is free, it happens inside YOLO's own NMS), while
    # 0.2/0.15 start pulling in enough extra boxes that they could plausibly
    # be noise rather than missed vehicles -- unverifiable without manual
    # ground-truth counts, so 0.25 is the conservative side of "don't miss
    # vehicles" rather than the aggressive side of "count everything."
    confidence_threshold: float = 0.25  # only used by detector_kind="yolo"
    iou_threshold: float = 0.45  # only used by detector_kind="yolo"
    # When True, roi_polygon/queue_polygon are fractions in [0, 1] of frame
    # width/height rather than absolute pixels -- resolution- and
    # orientation-independent by construction, so the same config works on
    # a portrait or landscape upload of any size. When False (the default,
    # preserved for every existing pixel-coordinate config/test), polygons
    # are absolute pixels as before.
    roi_normalized: bool = False


def scale_polygon(polygon: List[Point], frame_width: int, frame_height: int) -> List[Point]:
    """Converts a normalized ([0,1] fraction) polygon to absolute pixel
    coordinates for a frame of the given size. Used when ROIConfig.roi_normalized
    is True -- called once per resize (frame dimensions are stable per video)
    rather than assuming a fixed authoring resolution like 640x480."""
    return [(x * frame_width, y * frame_height) for x, y in polygon]


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """Standard ray-casting point-in-polygon test. No OpenCV dependency on
    purpose -- cv2.pointPolygonTest would work too, but this keeps ROI
    geometry testable without OpenCV installed."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def load_roi_configs(path: str) -> Dict[str, ROIConfig]:
    """Load a JSON file mapping direction -> ROI settings into ROIConfig
    objects. See backend/cv/configs/demo_single_direction.json for the
    expected shape."""
    data = json.loads(Path(path).read_text())
    configs: Dict[str, ROIConfig] = {}
    for direction, entry in data.items():
        roi_polygon = [tuple(p) for p in entry["roi_polygon"]]
        configs[direction] = ROIConfig(
            direction=direction,
            video_source=entry["video_source"],
            roi_polygon=roi_polygon,
            queue_polygon=[tuple(p) for p in entry.get("queue_polygon", roi_polygon)],
            frame_skip=entry.get("frame_skip", 2),
            resize_width=entry.get("resize_width", 640),
            detector_kind=entry.get("detector_kind", "yolo"),
            confidence_threshold=entry.get("confidence_threshold", 0.25),
            iou_threshold=entry.get("iou_threshold", 0.45),
            roi_normalized=entry.get("roi_normalized", False),
        )
    return configs

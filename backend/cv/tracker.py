"""Minimal centroid tracker: assigns stable IDs to detections across frames
by nearest-centroid matching, greedily assigned in order of increasing
distance.

Deliberately not a full Kalman/SORT tracker. Reference research (SORT's
canonical implementation is GPLv3 and pulls in filterpy/lap/scikit-image)
showed that's the wrong tradeoff for a demo that only needs counts and an
arrival rate, not motion prediction -- see the CV-research notes from the
milestone before this one. This gives "tracking IDs where practical"
without a GPL dependency or extra packages.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .detector import Detection

Point = Tuple[float, float]

# How many recent per-frame classifications a track remembers before
# reporting a class label. A small YOLO model genuinely flips class on
# ambiguous vehicle shapes (SUV/van vs. truck/bus) frame to frame even
# though it's obviously the same physical vehicle -- measured on real
# traffic footage during demo validation: 20 of 29 tracks (69%) flipped
# class at least once over their lifetime without this. Reporting the mode
# of a short recent window instead of the single latest frame's raw label
# removes that flicker while still tracking a real, sustained
# reclassification (it just takes a few frames to "catch up").
CLASS_SMOOTHING_WINDOW = 7


@dataclass
class Track:
    track_id: int
    centroid: Point
    class_name: str
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    confidence: float = 0.0
    velocity: Point = (0.0, 0.0)  # (dx, dy) per update, for an optional motion-direction readout
    hits: int = 1  # total number of detections this track has matched, ever
    age_since_seen: int = 0
    class_history: List[str] = field(default_factory=list)  # last CLASS_SMOOTHING_WINDOW raw classifications


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_age: int = 10) -> None:
        self.max_distance = max_distance
        self.max_age = max_age
        self._tracks: Dict[int, Track] = {}
        self._next_id = 0

    @property
    def tracks(self) -> Dict[int, Track]:
        return self._tracks

    def update(self, detections: List[Detection]) -> Dict[int, Track]:
        # Greedy nearest-first, one-to-one: each candidate (track, detection)
        # pair within max_distance is considered in increasing distance
        # order, and a track/detection index is removed from further
        # consideration the moment either side of it is matched -- so no
        # track can claim two detections and no detection can feed two
        # tracks, which is what keeps IDs from duplicating.
        candidate_pairs: List[Tuple[int, int, float]] = []
        for track_id, track in self._tracks.items():
            for i, det in enumerate(detections):
                dist = math.hypot(track.centroid[0] - det.centroid[0], track.centroid[1] - det.centroid[1])
                if dist <= self.max_distance:
                    candidate_pairs.append((track_id, i, dist))
        candidate_pairs.sort(key=lambda pair: pair[2])

        matched_track_ids: set = set()
        matched_detection_indices: set = set()
        for track_id, i, _dist in candidate_pairs:
            if track_id in matched_track_ids or i in matched_detection_indices:
                continue
            matched_track_ids.add(track_id)
            matched_detection_indices.add(i)
            det = detections[i]
            prev = self._tracks[track_id]
            velocity = (det.centroid[0] - prev.centroid[0], det.centroid[1] - prev.centroid[1])
            history = (prev.class_history + [det.class_name])[-CLASS_SMOOTHING_WINDOW:]
            smoothed_class = Counter(history).most_common(1)[0][0]
            self._tracks[track_id] = Track(
                track_id=track_id, centroid=det.centroid, class_name=smoothed_class,
                bbox=det.bbox, confidence=det.confidence, velocity=velocity, hits=prev.hits + 1,
                class_history=history,
            )

        for track_id in list(self._tracks.keys()):
            if track_id not in matched_track_ids:
                track = self._tracks[track_id]
                track.age_since_seen += 1
                if track.age_since_seen > self.max_age:
                    del self._tracks[track_id]

        for i, det in enumerate(detections):
            if i not in matched_detection_indices:
                self._tracks[self._next_id] = Track(
                    track_id=self._next_id, centroid=det.centroid, class_name=det.class_name,
                    bbox=det.bbox, confidence=det.confidence, class_history=[det.class_name],
                )
                self._next_id += 1

        return self._tracks

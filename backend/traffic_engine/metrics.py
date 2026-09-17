"""Shared data contracts between the CV/simulator layer and the agents.

Both the simulator and the future YOLO adapter must produce TrafficMetrics,
so the two sources stay interchangeable for every agent downstream.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficMetrics:
    direction: str
    vehicle_count: int  # active count: currently-tracked vehicles (coasts through brief misses)
    queue_length: int
    avg_wait_time: float
    arrival_rate: float
    timestamp: float
    density: float = 0.0  # vehicle_count normalized against an assumed lane capacity
    source_id: str = ""  # e.g. a video filename, camera index, or "simulator:<profile>"
    # CV-only diagnostics, additive so the simulator (which never sets them)
    # stays byte-for-byte compatible with every existing consumer.
    unique_vehicle_count: int = 0  # cumulative distinct track IDs ever seen this run
    detector_latency_ms: float = 0.0
    tracker_fps: float = 0.0


@dataclass(frozen=True)
class PriorityUpdate:
    direction: str
    priority_score: float
    vehicle_count: int
    queue_length: int
    avg_wait_time: float
    arrival_rate: float
    time_since_last_green: float
    is_starved: bool
    timestamp: float
    density: float = 0.0

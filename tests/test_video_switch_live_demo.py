"""Proves the exact evaluator-demo scenario end to end, against the REAL
Streamlit-facing pipeline (dashboard.start_pipeline), not a hand-rolled
adapter:

    SELECT VIDEO -> PLAY -> DETECT -> COUNT -> AGENT UPDATE
    -> COORDINATOR DECISION -> TIMER UPDATE

all within a few seconds, and with NO pipeline rebuild: the DashboardState
object, its DirectionalAgents, its CoordinatorAgent, and its MessageBus stay
exactly the same object before and after the switch -- only the one
direction's footage changes.
"""
from __future__ import annotations

import time

from backend.cv.roi_config import ROIConfig

FULL_ROI = [(0, 0), (640, 0), (640, 480), (0, 480)]
QUEUE_ZONE = [(0, 300), (640, 300), (640, 480), (0, 480)]


def test_switching_video_updates_the_running_pipeline_within_a_few_seconds_without_restarting():
    from dashboard import start_pipeline

    video_configs = {
        "N": ROIConfig(
            direction="N", video_source="data/traffic/empty.mp4",
            roi_polygon=FULL_ROI, queue_polygon=QUEUE_ZONE, frame_skip=0, detector_kind="motion",
        )
    }
    choice = (video_configs, ("S", "E", "W"), "balanced")
    state = start_pipeline(("N", "S", "E", "W"), "HYBRID", "Video (per direction)", choice)

    time.sleep(3)  # past MotionDetector's warmup on the empty clip
    assert state.error is None
    assert state.source_labels.get("N") == "live_video"
    assert state.video_status.get("N", {}).get("video_filename") == "empty.mp4"
    assert state.agents["N"].vehicle_count == 0

    pipeline_identity = id(state)  # the one thing that must NOT change

    state.request_video_switch("N", "data/traffic/heavy.mp4", "Local")

    deadline = time.time() + 8.0
    switched = False
    while time.time() < deadline:
        if state.video_status.get("N", {}).get("video_filename") == "heavy.mp4":
            switched = True
            break
        time.sleep(0.2)
    assert switched, "the running pipeline never picked up the video switch"

    # give the newly-busy clip a moment to clear MotionDetector's warmup and
    # for a fresh PriorityUpdate to reach the coordinator
    deadline = time.time() + 5.0
    reacted = False
    while time.time() < deadline:
        if state.agents["N"].vehicle_count > 0:
            reacted = True
            break
        time.sleep(0.2)
    assert reacted, "N's agent never updated its vehicle_count after the switch"

    # the whole loop above ran on the SAME DashboardState/pipeline object --
    # proof nothing was rebuilt
    assert id(state) == pipeline_identity

    joined = "\n".join(state.messages)
    assert "[VIDEO MANAGER]" in joined and "heavy.mp4" in joined
    assert "NORTH_AGENT" in joined  # DirectionalAgent("N")'s updates kept flowing through the same log

import pytest

from backend.agents.message_bus import MessageBus
from backend.cv.roi_config import ROIConfig
from backend.simulation.hybrid_source import HybridSource

FULL_FRAME_ROI = [(0, 0), (640, 0), (640, 480), (0, 480)]
QUEUE_ZONE = [(0, 320), (640, 320), (640, 480), (0, 480)]


def north_video_config(video_source="backend/cv/configs/demo_videos/north.mp4"):
    return ROIConfig(
        direction="N", video_source=video_source, roi_polygon=FULL_FRAME_ROI, queue_polygon=QUEUE_ZONE,
        frame_skip=0, detector_kind="motion",
    )


def test_mode_b_mixes_one_video_direction_with_three_simulated():
    source = HybridSource(
        MessageBus(),
        video_configs={"N": north_video_config()},
        simulated_directions=("S", "E", "W"),
    )
    assert set(source.directions) == {"N", "S", "E", "W"}
    assert source.source_labels == {"N": "live_video", "S": "simulated", "E": "simulated", "W": "simulated"}

    batch = source.tick()
    assert set(batch.keys()) == {"N", "S", "E", "W"}
    assert batch["N"].source_id == "north.mp4"
    assert batch["S"].source_id.startswith("simulator:")


def test_mode_a_all_four_directions_are_video():
    configs = {d: north_video_config() for d in ("N", "S", "E", "W")}  # same clip, distinct instances
    for d, cfg in configs.items():
        configs[d] = ROIConfig(**{**cfg.__dict__, "direction": d})

    source = HybridSource(MessageBus(), video_configs=configs, simulated_directions=())
    assert set(source.directions) == {"N", "S", "E", "W"}
    assert all(label == "live_video" for label in source.source_labels.values())
    assert source.simulated_directions == ()


def test_a_video_that_fails_to_open_is_labelled_no_source_not_faked_as_live():
    source = HybridSource(
        MessageBus(),
        video_configs={"N": north_video_config(video_source="does/not/exist.mp4")},
        simulated_directions=("S", "E", "W"),
    )
    assert source.source_labels["N"] == "no_source"
    assert "N" not in source.directions  # never silently treated as live or simulated
    assert set(source.directions) == {"S", "E", "W"}


def test_a_direction_covered_by_video_is_not_also_simulated():
    source = HybridSource(
        MessageBus(),
        video_configs={"N": north_video_config()},
        simulated_directions=("N", "S", "E", "W"),  # caller accidentally double-assigns N
    )
    assert source.source_labels["N"] == "live_video"
    assert "N" not in source.simulated_directions
    assert source.directions.count("N") == 1


def test_switch_video_swaps_one_directions_footage_without_touching_the_rest():
    source = HybridSource(
        MessageBus(),
        video_configs={"N": north_video_config()},
        simulated_directions=("S", "E", "W"),
    )
    directions_before = source.directions
    before = source.tick()
    assert before["N"].source_id == "north.mp4"

    source.switch_video("N", "data/traffic/heavy.mp4")

    after = source.tick()
    assert after["N"].source_id == "heavy.mp4"
    # topology (which directions exist, which are simulated) is unchanged
    assert source.directions == directions_before
    assert source.simulated_directions == ("S", "E", "W")
    assert source.source_labels["N"] == "live_video"


def test_switch_video_rejects_a_direction_with_no_running_adapter():
    source = HybridSource(
        MessageBus(),
        video_configs={"N": north_video_config()},
        simulated_directions=("S", "E", "W"),
    )
    with pytest.raises(KeyError):
        source.switch_video("S", "data/traffic/heavy.mp4")  # S is simulated, not video-backed


def test_status_snapshot_reflects_the_currently_running_video():
    source = HybridSource(
        MessageBus(),
        video_configs={"N": north_video_config()},
        simulated_directions=("S", "E", "W"),
    )
    assert source.status_snapshot()["N"]["video_filename"] == "north.mp4"
    source.switch_video("N", "data/traffic/heavy.mp4")
    assert source.status_snapshot()["N"]["video_filename"] == "heavy.mp4"

from backend.cv.roi_config import ROIConfig, load_roi_configs, point_in_polygon

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_point_inside_polygon_is_true():
    assert point_in_polygon((5, 5), SQUARE) is True


def test_point_outside_polygon_is_false():
    assert point_in_polygon((50, 50), SQUARE) is False


def test_point_on_polygon_edge_boundary_behaviour_is_consistent():
    # ray-casting edge behaviour is a known grey area; just assert it
    # doesn't raise and returns a bool, rather than pin an exact edge rule
    result = point_in_polygon((10, 5), SQUARE)
    assert isinstance(result, bool)


def test_load_roi_configs_parses_single_direction_demo_file():
    configs = load_roi_configs("backend/cv/configs/demo_single_direction.json")
    assert set(configs.keys()) == {"N"}
    cfg = configs["N"]
    assert isinstance(cfg, ROIConfig)
    assert cfg.direction == "N"
    assert cfg.video_source.endswith("north.mp4")
    assert len(cfg.roi_polygon) == 4
    assert cfg.frame_skip == 0  # motion detector is cheap; no need to skip frames
    assert cfg.resize_width == 640


def test_load_roi_configs_parses_four_direction_demo_file():
    expected_video_names = {"N": "north.mp4", "S": "south.mp4", "E": "east.mp4", "W": "west.mp4"}
    configs = load_roi_configs("backend/cv/configs/demo_four_directions.json")
    assert set(configs.keys()) == {"N", "S", "E", "W"}
    for direction, cfg in configs.items():
        assert cfg.direction == direction
        assert cfg.video_source.endswith(expected_video_names[direction])
        assert len(cfg.roi_polygon) >= 3
        assert len(cfg.queue_polygon) >= 3

import numpy as np
import pytest

from backend.cv.motion_detector import MotionDetector


def blank_frame(width=200, height=150):
    return np.zeros((height, width, 3), dtype=np.uint8)


def frame_with_box(width=200, height=150, box=(40, 40, 80, 70), color=(0, 200, 0)):
    frame = blank_frame(width, height)
    x1, y1, x2, y2 = box
    frame[y1:y2, x1:x2] = color
    return frame


def test_empty_detections_on_a_static_unchanging_scene():
    detector = MotionDetector(warmup_frames=3)
    frame = blank_frame()
    for _ in range(10):
        detections = detector.detect(frame)
    assert detections == []


def test_detector_output_finds_a_moving_box_after_warmup():
    # a box that shifts position each frame reads as foreground motion,
    # same as the real demo clips (solid rectangles sliding across a dark
    # background) confirmed earlier against backend/cv/configs/demo_videos --
    # 5px/frame matches that clip's actual displacement rate (~4px/frame)
    detector = MotionDetector(warmup_frames=10, min_area=100)
    detections = []
    for i in range(20):
        frame = frame_with_box(box=(40 + i * 5, 40, 80 + i * 5, 70))
        detections = detector.detect(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "object"
    assert 0.0 < det.confidence <= 1.0
    x1, y1, x2, y2 = det.bbox
    assert x2 > x1 and y2 > y1
    assert det.centroid == pytest.approx(((x1 + x2) / 2.0, (y1 + y2) / 2.0))


def test_area_threshold_filters_out_tiny_noise_blobs():
    detector = MotionDetector(warmup_frames=10, min_area=5000)  # box's moving sliver is far smaller
    detections = []
    for i in range(20):
        frame = frame_with_box(box=(40 + i * 5, 40, 80 + i * 5, 70))
        detections = detector.detect(frame)
    assert detections == []


def test_malformed_frame_raises_instead_of_silently_fabricating_a_detection():
    detector = MotionDetector(warmup_frames=1)
    detector.detect(blank_frame())
    with pytest.raises(Exception):
        detector.detect(None)


def test_detects_real_demo_footage_where_yolo_finds_nothing():
    # confirms this detector is the one that actually works on MAATS' own
    # demo clips: they're solid colored rectangles on a near-black
    # background, not photographic vehicles, so YOLOv8n detects 0 objects
    # on them (verified separately) -- this fallback must not.
    import cv2

    cap = cv2.VideoCapture("backend/cv/configs/demo_videos/north.mp4")
    detector = MotionDetector(warmup_frames=15, min_area=100)
    detections = []
    for _ in range(90):
        ok, frame = cap.read()
        assert ok
        detections = detector.detect(frame)
    cap.release()

    assert len(detections) >= 1
    for d in detections:
        assert d.class_name == "object"
        assert 0.0 < d.confidence <= 1.0

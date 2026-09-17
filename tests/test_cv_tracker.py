from backend.cv.detector import Detection
from backend.cv.tracker import CentroidTracker


def det(x1, y1, x2, y2, class_name="car", confidence=0.9):
    return Detection(class_name=class_name, confidence=confidence, bbox=(x1, y1, x2, y2))


def test_new_detection_gets_a_new_track_id():
    tracker = CentroidTracker()
    tracks = tracker.update([det(0, 0, 10, 10)])
    assert len(tracks) == 1


def test_same_vehicle_moving_slightly_keeps_the_same_id():
    tracker = CentroidTracker(max_distance=50.0)
    tracks = tracker.update([det(0, 0, 10, 10)])
    (first_id,) = tracks.keys()

    tracks = tracker.update([det(5, 5, 15, 15)])  # centroid moved by ~7px
    assert set(tracks.keys()) == {first_id}


def test_vehicle_far_away_gets_a_new_id_instead_of_matching():
    tracker = CentroidTracker(max_distance=20.0)
    tracker.update([det(0, 0, 10, 10)])
    tracks = tracker.update([det(500, 500, 510, 510)])
    assert len(tracks) == 2  # far away: treated as a different vehicle, old one aged out later


def test_track_is_dropped_after_max_age_of_no_detections():
    tracker = CentroidTracker(max_distance=50.0, max_age=2)
    tracker.update([det(0, 0, 10, 10)])
    tracker.update([])  # miss 1
    tracker.update([])  # miss 2
    tracks = tracker.update([])  # miss 3 -> should be dropped
    assert tracks == {}


def test_two_close_detections_are_assigned_by_nearest_first():
    tracker = CentroidTracker(max_distance=100.0)
    tracker.update([det(0, 0, 10, 10), det(100, 100, 110, 110)])
    ids_round1 = set(tracker.tracks.keys())

    # both move slightly; nearest-neighbour greedy assignment should keep
    # each track attached to the detection closest to its last position
    tracks = tracker.update([det(102, 102, 112, 112), det(2, 2, 12, 12)])
    assert set(tracks.keys()) == ids_round1


def test_two_new_detections_near_one_track_do_not_duplicate_its_id():
    # both candidate detections are within max_distance of the one existing
    # track -- exactly one may claim it (nearest), the other must become a
    # genuinely new track, never a duplicate of the same ID
    tracker = CentroidTracker(max_distance=100.0)
    tracker.update([det(0, 0, 10, 10)])
    (original_id,) = tracker.tracks.keys()

    tracks = tracker.update([det(5, 5, 15, 15), det(8, 8, 18, 18)])
    assert len(tracks) == 2
    assert len(set(tracks.keys())) == 2  # no duplicate IDs
    assert original_id in tracks  # the nearer detection kept the original ID


def test_track_survives_a_temporary_miss_and_recovers_the_same_id():
    tracker = CentroidTracker(max_distance=50.0, max_age=3)
    tracker.update([det(0, 0, 10, 10)])
    (original_id,) = tracker.tracks.keys()

    tracks_during_miss = tracker.update([])  # one frame with no detections
    assert original_id in tracks_during_miss  # still coasting, not dropped

    tracks_after_recovery = tracker.update([det(3, 3, 13, 13)])  # vehicle reappears nearby
    assert set(tracks_after_recovery.keys()) == {original_id}  # same ID, not a new one


def test_confidence_bbox_and_class_are_exposed_per_track():
    tracker = CentroidTracker()
    tracks = tracker.update([det(0, 0, 10, 10, class_name="bus", confidence=0.77)])
    (track,) = tracks.values()
    assert track.class_name == "bus"
    assert track.confidence == 0.77
    assert track.bbox == (0, 0, 10, 10)


def test_class_label_is_smoothed_against_a_single_frame_flip():
    # Regression test for a real defect found during demo validation on
    # actual traffic footage: a small YOLO model genuinely flips class on
    # ambiguous vehicle shapes frame to frame even for the same physical
    # vehicle (measured: 20/29 tracks flipped class at least once on a real
    # video). One single-frame misclassification must not change the
    # track's reported class -- the mode of recent history should win.
    tracker = CentroidTracker(max_distance=50.0)
    x = 0
    for _ in range(5):
        tracker.update([det(x, 0, x + 10, 10, class_name="car")])
        x += 2
    (track_id,) = tracker.tracks.keys()
    assert tracker.tracks[track_id].class_name == "car"

    # one single-frame misclassification (e.g. "bus") must not flip the
    # reported class away from the well-established "car"
    tracks = tracker.update([det(x, 0, x + 10, 10, class_name="bus")])
    assert tracks[track_id].class_name == "car"


def test_class_label_does_track_a_sustained_reclassification():
    # The smoothing window must not be permanently "stuck" -- a genuinely
    # sustained new classification should eventually win once it dominates
    # the recent window.
    tracker = CentroidTracker(max_distance=50.0)
    x = 0
    for _ in range(3):
        tracker.update([det(x, 0, x + 10, 10, class_name="car")])
        x += 2
    (track_id,) = tracker.tracks.keys()

    tracks = None
    for _ in range(6):
        tracks = tracker.update([det(x, 0, x + 10, 10, class_name="truck")])
        x += 2
    assert tracks[track_id].class_name == "truck"

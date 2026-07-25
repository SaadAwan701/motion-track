import pytest
import numpy as np
from motion_track.tracker import CentroidTracker


def test_centroid_tracker_lifecycle():
    tracker = CentroidTracker(max_disappeared=2)

    # Frame 1: Single object detected at box (10, 10, 20, 20) --> Centroid: (20, 20)
    rects_f1 = [(10, 10, 20, 20)]
    objects = tracker.update(rects_f1)

    assert len(objects) == 1
    assert 0 in objects
    np.testing.assert_array_equal(objects[0], [20, 20])

    # Frame 2: Object moves slightly to (12, 12, 20, 20) --> Centroid: (22, 22)
    rects_f2 = [(12, 12, 20, 20)]
    objects = tracker.update(rects_f2)

    assert len(objects) == 1
    assert 0 in objects
    np.testing.assert_array_equal(objects[0], [22, 22])

    # Frames 3, 4, 5: Object disappears completely
    tracker.update([])
    tracker.update([])
    objects = tracker.update([])  # Surpasses max_disappeared (2)

    assert len(objects) == 0

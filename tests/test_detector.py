import pytest
import numpy as np
import cv2
from motion_track.detector import MotionDetector


def test_motion_detector_detects_movement():
    baseline_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    current_frame = baseline_frame.copy()

    cv2.rectangle(current_frame, (50, 50), (120, 120), (255, 255, 255), -1)

    detector = MotionDetector(min_area=100)
    base_preprocess = detector.preprocess(baseline_frame)
    curr_preprocess = detector.preprocess(current_frame)

    boxes, mask = detector.detect(base_preprocess, curr_preprocess)

    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert w > 0
    assert h > 0

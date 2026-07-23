import cv2
import numpy as np


class MotionDetector:
    def __init__(self, threshold_val=25, min_area=500, blur_kernel=(21, 21)):
        self.min_area = min_area
        self.threshold_val = threshold_val
        self.blur_kernel = blur_kernel

    def preprocess(self, frame):
        gscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gscale_img, self.blur_kernel, 0)
        return blurred_img

    def detect(self, baseline_frame: np.ndarray, current_frame: np.ndarray):
        delta_img = cv2.absdiff(baseline_frame, current_frame)
        _, thresh = cv2.threshold(delta_img, self.threshold_val, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(thresh, None, iterations=2)  # type: ignore
        ct, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for i in ct:
            ct_area = cv2.contourArea(i)
            if ct_area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(i)
            bounding_boxes.append((x, y, w, h))

        return (bounding_boxes, dilated)

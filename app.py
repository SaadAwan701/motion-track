import cv2
import argparse
from motion_track.detector import MotionDetector
from motion_track.tracker import CentroidTracker


def main():

    parser = argparse.ArgumentParser(
        description="Real-time Motion Detection & Tracking"
    )
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default=None,
        help="Path to input video file (default: webcam 0)",
    )
    args = parser.parse_args()

    src = args.video if args.video else 0
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Could not open video source '{src}'.")
        return

    # setting up the detector and tracker
    detector = MotionDetector(min_area=500, threshold_val=25)
    tracker = CentroidTracker(max_disappeared=40)

    print("Starting video stream. Press 'q' to exit.")

    #  Getting the first frame for baseline comparison
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read initial frame.")
        cap.release()
        return

    baseline_prep = detector.preprocess(first_frame)

    while True:
        ret, current_frame = cap.read()
        if not ret:
            print("End of video stream reached.")
            break

        # Preprocess current frame & run motion detector
        current_prep = detector.preprocess(current_frame)
        boxes, mask = detector.detect(baseline_prep, current_prep)

        # Update centroid tracker with detected bounding boxes
        objects = tracker.update(boxes)

        # Drawing bounding boxes from detector and then centroids from traker
        for x, y, w, h in boxes:
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for object_id, centroid in objects.items():
            text = f"ID {object_id}"
            c_x, c_y = centroid[0], centroid[1]

            # drawin the centroid
            cv2.circle(current_frame, (c_x, c_y), 4, (0, 0, 255), -1)
            cv2.putText(
                current_frame,
                text,
                (c_x - 10, c_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        # Display output windows
        cv2.imshow("Motion Track - Main Feed", current_frame)
        cv2.imshow("Motion Track - Dilated Mask", mask)

        baseline_prep = current_prep

        # Break loop on pressing 'q'
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

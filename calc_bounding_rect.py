import cv2
import numpy as np
import pickle
import argparse

from video_capture_threading import VideoCaptureThreading as VideoCapture

color_calib_hsvs = []

for ind in range(1, 4):
    with open(f"hsv_calibration_data_{ind}.pkl", "rb") as f:
        color_calib_hsvs.append(pickle.load(f))


def preprocess_frame(frame):
    # Convert to HSV
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sharpen
    # frame_sharpen = cv2.GaussianBlur(hsv_img, (0, 0), 1)
    # frame_sharpen = cv2.addWeighted(hsv_img, 1.5, frame_sharpen, -0.5, 0)

    return hsv_img


def threshold_for_color(hsv_img, color, calib_num=1):
    [color_min_hsv, color_max_hsv] = color_calib_hsvs[calib_num-1]

    (low_H, low_S, low_V) = color_min_hsv[color]
    (high_H, high_S, high_V) = color_max_hsv[color]
    if low_H > high_H:
        # Wrap around
        frame_threshold = cv2.bitwise_or(
            cv2.inRange(hsv_img, (low_H, low_S, low_V), (360, high_S, high_V)),
            cv2.inRange(hsv_img, (0, low_S, low_V), (high_H, high_S, high_V)),
        )
    else:
        frame_threshold = cv2.inRange(
            hsv_img, (low_H, low_S, low_V), (high_H, high_S, high_V)
        )

    morph_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, morph_ellipse)
    return frame_threshold


def remove_fiducials(rgb_img, mask):
    # Dialate mask
    mask_dilated = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=1
    )

    return cv2.inpaint(rgb_img, mask_dilated, 10, cv2.INPAINT_NS)


def detect_squares(threshold_img):
    # Returns a list of cv2.RotatedRect

    # Use OpenCV to find contours
    contours, _ = cv2.findContours(
        threshold_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter squares by area and aspect ratio
    squares = [cv2.minAreaRect(c) for c in contours if cv2.contourArea(c) > 30]
    squares = [s for s in squares if 0.3 <= s[1][0] / s[1][1] <= 1.8]
    # squares = [cv2.boxPoints(sq) for sq in squares]

    return squares


def convert_fiducial_squares_to_bounding_rect(squares):
    if len(squares) != 4:
        # Bad frame, skip it
        return None

    # Sort squares by x-coordinate
    squares.sort(key=lambda s: s[0][0])

    # Sort first two squares by y-coordinate, and second two squares independently
    squares = [
        *sorted(list(squares[0:2]), key=lambda s: s[0][1]),
        *sorted(list(squares[2:4]), key=lambda s: s[0][1]),
    ]

    squares = [list(cv2.boxPoints(sq)) for sq in squares]
    for sq in squares:
        sq.sort(key=lambda p: p[0])
    squares = [
        [
            *sorted(list(sq[0:2]), key=lambda s: s[1]),
            *sorted(list(sq[2:4]), key=lambda s: s[1]),
        ]
        for sq in squares
    ]

    bounding_rect = np.array(
        [
            squares[0][3],
            squares[1][2],
            squares[3][0],
            squares[2][1],
        ]
    )

    return bounding_rect


def draw_squares(frame, squares):
    for sq in squares:
        draw_square(frame, sq)


def draw_square(frame, sq):
    cv2.drawContours(frame, [np.int0(sq)], 0, (100, 100, 255), 3)


if __name__ == "__main__":
    # Command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cam_port", "-p", type=str, default=0, help="OpenCV camera port or video file"
    )
    parser.add_argument(
        "--cap_width", "-x", type=int, default=3840, help="Camera capture width"
    )
    parser.add_argument(
        "--cap_height", "-y", type=int, default=2160, help="Camera capture height"
    )
    parser.add_argument(
        "--cap_fps", "-f", type=int, default=30, help="Camera capture FPS"
    )
    parser.add_argument(
        "--cam_calib",
        "-c",
        type=str,
        default="camera_calibration_data.pkl",
        help="Camera calibration",
    )
    parser.add_argument("--use_calib", "-u", action="store_true")
    args = parser.parse_args()

    # Read frames from webcam
    if args.cam_port.isdigit():
        cap = VideoCapture(
            port=int(args.cam_port),
            width=args.cap_width,
            height=args.cap_height,
            fps=args.cap_fps,
            calib=args.cam_calib,
        ).start()
    else:
        cap = cv2.VideoCapture(args.cam_port)

    while True:
        # Read frame
        ret, frame = cap.read_calib() if args.use_calib else cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        hsv_img = preprocess_frame(frame)

        rects = []

        for k in ["green"]:
            thresh_img = threshold_for_color(hsv_img, k)
            # frame = remove_fiducials(frame, thresh_img)

            # Find contours
            squares = detect_squares(thresh_img)
            rect = convert_fiducial_squares_to_bounding_rect(squares)
            if rect is not None:
                rects.append(rect)

        for rect in rects:
            draw_square(frame, rect)

        cv2.imshow("frame", frame)

        # Check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Stop the camera
    cv2.destroyAllWindows()

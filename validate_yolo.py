import cv2
import os
import yolov5

from calc_bounding_rect import (
    preprocess_frame,
    threshold_for_color,
    detect_squares,
    convert_fiducial_squares_to_bounding_rect,
    remove_fiducials,
)

DATASET_NAME = "breaker"
WEIGHTS = "runs/train/exp6/weights/best.pt"

if __name__ == "__main__":
    # Load model
    model = yolov5.load(WEIGHTS)

    # Load validation videos
    video_list = []

    for root, subdirs, files in os.walk(os.path.join("data", DATASET_NAME, "val")):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_name, file_ext = os.path.splitext(file_path)

            # If the file is an video...
            if file_ext == ".MOV":
                # Add to list
                video_list.append((file_name, file_ext))

    # For each video...
    for video_path, video_ext in video_list:
        video = cv2.VideoCapture(video_path + video_ext)

        # For each frame...
        while True:
            # Read frame
            ret, frame = video.read()

            # If the frame is empty...
            if frame is None:
                break

            hsv_img = preprocess_frame(frame)
            thresh_img = threshold_for_color(hsv_img, "fiducial_yellow")
            squares = detect_squares(thresh_img)
            rect = convert_fiducial_squares_to_bounding_rect(squares)
            frame = remove_fiducials(frame, thresh_img)

            results = model(frame)
            imgs = results.render()

            # Display frame
            cv2.imshow("Frame", imgs[0])
            key = cv2.waitKey(1)

            # If the user presses the 'q' key...
            if key == ord("q"):
                break

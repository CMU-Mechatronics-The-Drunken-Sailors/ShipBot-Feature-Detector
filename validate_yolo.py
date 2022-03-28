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

DATASET_NAME = "shipbot"
# WEIGHTS = "good_weights/500_epochs_both_panels.pt"
WEIGHTS = "runs/train/exp/weights/best.pt"

if __name__ == "__main__":
    # Load model
    model = yolov5.load(WEIGHTS)

    # Load validation videos
    video_list = []

    # root = os.path.join("data", DATASET_NAME, "val")
    root = os.path.join("data", DATASET_NAME)
    for filename in os.listdir(root):
        file_path = os.path.join(root, filename)
        file_name, file_ext = os.path.splitext(file_path)

        # If the file is an video...
        if (file_ext == ".MOV" or file_ext == ".mp4") and "SPIGOTTOPVIEW" in file_name:
            # Add to list
            video_list.append((file_name, file_ext))

    # For each video...
    for video_path, video_ext in video_list:
        video = cv2.VideoCapture(video_path + video_ext)

        # For each frame...
        while True:
            for _ in range(10):
                video.grab()

            # Read frame
            ret, frame = video.read()

            # If the frame is empty...
            if frame is None:
                break

            # Resize frame to 640x480
            frame = cv2.resize(frame, (640, 360))

            results = model(frame)
            imgs = results.render()

            # Display frame
            cv2.imshow("Frame", imgs[0])
            key = cv2.waitKey(1)

            # If the user presses the 'q' key...
            if key == ord("q"):
                break

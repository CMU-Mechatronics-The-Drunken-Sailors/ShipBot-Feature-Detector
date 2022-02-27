import cv2
import os
from multiprocessing import Pool
import yaml

DATASET_NAME = "breaker"

from calc_bounding_rect import (
    preprocess_frame,
    threshold_for_color,
    detect_squares,
    convert_fiducial_squares_to_bounding_rect,
    remove_fiducials,
)

labels_list = yaml.load(
    open(os.path.join("data", DATASET_NAME, f"{DATASET_NAME}.yaml")),
    Loader=yaml.SafeLoader,
)["names"]


def process_video(file_name, file_ext):
    # Read video
    cap = cv2.VideoCapture(file_name + file_ext)

    # Get label
    dir = os.path.dirname(file_name)
    label = os.path.basename(file_name).split("_")[0]
    label_idx = labels_list.index(label)

    # For each frame...
    frame_ind = 0
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        # If frame is empty, break
        if not ret:
            break

        hsv_img = preprocess_frame(frame)
        thresh_img = threshold_for_color(hsv_img, "fiducial_yellow")
        frame = remove_fiducials(frame, thresh_img)
        squares = detect_squares(thresh_img)
        rect = convert_fiducial_squares_to_bounding_rect(squares)

        if rect is not None:
            # Convert rect from x1,y1,x2,y2 to YOLO coords (center x, center y, width, height)
            x, y, w, h = cv2.boundingRect(rect)
            x = x + w / 2
            y = y + h / 2

            # Normalize to [0,1]
            x = x / frame.shape[1]
            y = y / frame.shape[0]
            w = w / frame.shape[1]
            h = h / frame.shape[0]

            # Write to file
            with open(os.path.join(dir, f"{label}_{frame_ind}.txt"), "w") as f:
                f.write(f"{label_idx} {x} {y} {w} {h}")

            # Write image to file
            cv2.imwrite(os.path.join(dir, f"{label}_{frame_ind}.jpg"), frame)

            frame_ind += 1

    cap.release()


# For each image in the data folder...
if __name__ == "__main__":
    video_list = []

    for subdir in ["train", "test"]:
        for root, subdirs, files in os.walk(os.path.join("data", DATASET_NAME, subdir)):
            for filename in files:
                file_path = os.path.join(root, filename)
                file_name, file_ext = os.path.splitext(file_path)

                # If the file is an video...
                if file_ext == ".MOV":
                    # Add to list
                    video_list.append((file_name, file_ext))

        # Process images in parallel
        with Pool() as pool:
            pool.starmap(process_video, video_list)

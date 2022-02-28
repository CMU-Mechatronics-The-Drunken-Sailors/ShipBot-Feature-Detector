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


def process_video(idx, file_name, file_ext):
    # Read video
    cap = cv2.VideoCapture(file_name + file_ext)

    # Get label
    dir = os.path.dirname(file_name)
    labels = os.path.basename(file_name).split("_")[0:3]
    label_idxs = [labels_list.index(label) for label in labels]
    colors = ["pink", "red", "green"]

    # For each frame...
    frame_ind = 0
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        # If frame is empty, break
        if not ret:
            break

        out_str = ""
        hsv_img = preprocess_frame(frame)
        skip_frame = False

        for label, label_idx, color in zip(labels, label_idxs, colors):
            thresh_img = threshold_for_color(hsv_img, color)
            frame = remove_fiducials(frame, thresh_img)
            squares = detect_squares(thresh_img)
            rect = convert_fiducial_squares_to_bounding_rect(squares)

            if rect is None:
                skip_frame = True
                break
            else:
                # Convert rect from x1,y1,x2,y2 to YOLO coords (center x, center y, width, height)
                x, y, w, h = cv2.boundingRect(rect)
                x = x + w / 2
                y = y + h / 2

                # Normalize to [0,1]
                x = x / frame.shape[1]
                y = y / frame.shape[0]
                w = w / frame.shape[1]
                h = h / frame.shape[0]

                out_str += f"{label_idx} {x:.4f} {y:.4f} {w:.4f} {h:.4f}\n"

        if not skip_frame:
            final_dir = os.path.join(dir, "test" if frame_ind % 4 == 0 else "train")

            # Write to file
            with open(os.path.join(final_dir, f"{idx}_{frame_ind}.txt"), "w") as f:
                f.write(out_str)

            # Write image to file
            cv2.imwrite(os.path.join(final_dir, f"{idx}_{frame_ind}.jpg"), frame)

        frame_ind += 1

    cap.release()


# For each image in the data folder...
if __name__ == "__main__":
    video_list = []

    idx = 0
    for root, subdirs, files in os.walk(os.path.join("data", DATASET_NAME)):
        for filename in files:

            file_path = os.path.join(root, filename)
            file_name, file_ext = os.path.splitext(file_path)

            # If the file is an video...
            if file_ext == ".MOV" or file_ext == ".mp4":
                # Add to list
                video_list.append((idx, file_name, file_ext))
                idx += 1

    # Process images in parallel
    with Pool() as pool:
        pool.starmap(process_video, video_list)

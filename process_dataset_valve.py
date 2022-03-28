import cv2
import os
from multiprocessing import Pool
import yaml
import pickle

DATASET_NAME = "shipbot"

labels_list = yaml.load(
    open(os.path.join("data", DATASET_NAME, f"{DATASET_NAME}.yaml")),
    Loader=yaml.SafeLoader,
)["names"]

color_calib_hsvs = {}

with open(f"hsv_calibration_data_valve.pkl", "rb") as f:
    [color_min_hsv, color_max_hsv] = pickle.load(f)

def threshold_for_color(hsv_img, color):

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

    morph_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, morph_ellipse)
    return frame_threshold


def process_video(idx, file_name, file_ext):
    # Read video
    cap = cv2.VideoCapture(file_name + file_ext)

    # Get label
    dir = os.path.dirname(file_name)
    labels = os.path.basename(file_name).split("-")[0].split("_")
    label_idxs = [labels_list.index(label) for label in labels]

    video_ind = int(os.path.basename(file_name).split("-")[-1])

    # For each frame...
    frame_ind = 0
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        # If frame is empty, break
        if not ret:
            break

        # Resize to 720p
        frame = cv2.resize(frame, (1280, 720))

        out_str = ""
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skip_frame = False

        for label, label_idx in zip(labels, label_idxs):
            color = "blue_liberal" if label in ["STOPCOCKSIDEVIEW", "SPIGOTSIDEVIEW"] else "blue"

            thresh_img = threshold_for_color(hsv_img, color)

            # Find contours
            contours, _ = cv2.findContours(
                thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                # No contours
                skip_frame = True
                break

            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            rect = cv2.boundingRect(largest_contour)

            if rect is None:
                skip_frame = True
                break
            else:
                # Convert rect from x1,y1,x2,y2 to YOLO coords (center x, center y, width, height)
                x, y, w, h = rect
                x = x + w / 2
                y = y + h / 2

                # Normalize to [0,1]
                x = x / frame.shape[1]
                y = y / frame.shape[0]
                w = w / frame.shape[1]
                h = h / frame.shape[0]

                out_str += f"{label_idx} {x:.4f} {y:.4f} {w:.4f} {h:.4f}\n"

        if not skip_frame:
            final_dir = os.path.join(dir, "test" if frame_ind % 6 == 0 else "train")

            # Write to file
            with open(os.path.join(final_dir, f"valve_{idx}_{frame_ind}.txt"), "w") as f:
                f.write(out_str)

            # Write image to file
            cv2.imwrite(os.path.join(final_dir, f"valve_{idx}_{frame_ind}.jpg"), frame)

            # # Show rect
            # cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

            # # Show frame
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)

        frame_ind += 1

    cap.release()


# For each image in the data folder...
if __name__ == "__main__":
    video_list = []

    idx = 0
    root = os.path.join("data", DATASET_NAME)
    for filename in os.listdir(root):
        file_path = os.path.join(root, filename)
        file_name, file_ext = os.path.splitext(file_path)

        # If the file is an video...
        if (file_ext == ".MOV" or file_ext == ".mp4") and "BREAKER" not in file_name:
            # Add to list
            video_list.append((idx, file_name, file_ext))
            idx += 1

    # Process images in parallel
    with Pool() as pool:
        pool.starmap(process_video, video_list)

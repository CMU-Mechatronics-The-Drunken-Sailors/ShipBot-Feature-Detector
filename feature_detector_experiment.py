import cv2
import os
from multiprocessing import Pool
import yaml
import numpy as np

DATASET_NAME = "shipbot"

labels_list = yaml.load(
    open(os.path.join("data", DATASET_NAME, f"{DATASET_NAME}.yaml")),
    Loader=yaml.SafeLoader,
)["names"]

# Configure feature detector
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load template
spigottopviewtemplate = [cv2.imread(f"spigottopviewtemplate-{k}.png") for k in [1, 2]]
spigottopviewtemplate.insert(0, None) # Make the array 1-indexed

spigottopviewtemplate_kp = []
spigottopviewtemplate_des = []
for k in [1, 2]:
    kp, des = orb.detectAndCompute(spigottopviewtemplate[k], None)
    spigottopviewtemplate_kp.append(kp)
    spigottopviewtemplate_des.append(des)

spigottopviewtemplate_kp.insert(0, None) # Make the arrays 1-indexed
spigottopviewtemplate_des.insert(0, None)


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

        out_str = ""
        skip_frame = False

        for label, label_idx in zip(labels, label_idxs):
            rect = None
            if label == "SPIGOTTOPVIEW":
                kp2, des2 = orb.detectAndCompute(frame, None)

                matches = bf.match(spigottopviewtemplate_des[video_ind], des2)

                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)

                good_matches = matches[:100]

                src_pts = np.float32(
                    [spigottopviewtemplate_kp[video_ind][m.queryIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in good_matches]
                ).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w = spigottopviewtemplate[video_ind].shape[:2]
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
                dst += (w, 0)  # adding offset

                draw_params = dict(
                    matchColor=(0, 255, 0),  # draw matches in green color
                    singlePointColor=None,
                    matchesMask=matchesMask,  # draw only inliers
                    flags=2,
                )

                img3 = cv2.drawMatches(
                    spigottopviewtemplate[video_ind],
                    spigottopviewtemplate_kp[video_ind],
                    frame,
                    kp2,
                    good_matches,
                    None,
                    **draw_params,
                )
                img3 = cv2.polylines(
                    img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA
                )

                cv2.imshow("img3", img3)
                cv2.waitKey(1)

            if rect is None:
                skip_frame = True
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
            final_dir = os.path.join(dir, "test" if frame_ind % 6 == 0 else "train")

            # # Write to file
            # with open(os.path.join(final_dir, f"{idx}_{frame_ind}.txt"), "w") as f:
            #     f.write(out_str)

            # # Write image to file
            # cv2.imwrite(os.path.join(final_dir, f"{idx}_{frame_ind}.jpg"), frame)

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
        if (file_ext == ".MOV" or file_ext == ".mp4") and "SPIGOTTOPVIEW" in file_name:
            # Add to list
            video_list.append((idx, file_name, file_ext))
            idx += 1

    # Process images in parallel
    with Pool() as pool:
        pool.starmap(process_video, video_list)

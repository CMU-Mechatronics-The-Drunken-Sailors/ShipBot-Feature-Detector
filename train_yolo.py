import os
from yolov5 import train

DATASET_NAME = "breaker"

if __name__ == "__main__":
    train.run(
        imgsz=1280,
        data=os.path.join("data", DATASET_NAME, f"{DATASET_NAME}.yaml"),
        epochs=30,
        batch=16,
        weights="yolov5m.pt",
    )

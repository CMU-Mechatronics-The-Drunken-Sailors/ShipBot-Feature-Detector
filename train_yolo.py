import os
from yolov5 import train

DATASET_NAME = "breaker"

if __name__ == "__main__":
    train.run(
        imgsz=640,
        data=os.path.join("data", DATASET_NAME, f"{DATASET_NAME}.yaml"),
        epochs=500,
        batch=128,
        weights="yolov5m.pt",
    )

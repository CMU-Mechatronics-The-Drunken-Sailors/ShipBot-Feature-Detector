# ShipBot Feature Detector
> Use YOLO to identify circuit breakers and valves

## Instructions to run:
```bash
# You MUST use Python 3.9
pip install -r requirements.txt

# Preprocess data...
python hsv_calib.py # Calibrate fiducial detector
python calc_bounding_rect.py # Make sure we're finding the fiducials correctly and drawing good bounding rects
python process_dataset.py # Generate the dataset from the videos!

# To train...
python train_yolo.py

# To validate...
python validate_yolo.py # Be sure to modify WEIGHTS
```

## Notes

**Video format**: `data/(name of dataset)/[train, test, val]/(label name)_whatever_you_want_here.MOV`
**Train on GPU** for a huge speedup! Otherwise switch to a smaller model like `yolov5s.pt`.
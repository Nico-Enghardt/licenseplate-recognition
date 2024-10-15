# Readme

## Overview

Purspose of this code is to process the image and extraxt the license plate number. We are looking for Spanish license plates, therefore the format is [DDDDCCC], where D = digit, C = character.

Our main program is divided into two phases. 

### Phase 1: Detecting License Plate ROI (Region of Interest)

Detecting license plate ROI is being done with trained by ourselves model of yolov5

### Phase 2: OCR (Optical Character Recognition)

After test with EasyOCR, PyTesseract and PaddleOCR we have decided for PaddleOCR. It had the best success ratio and was effective even in dark condition. Works well without additional license plate image preprocessing, even in dark conditions.

## Installation

In the beginning you need to make sure that you have yolov5 downloaded and installed all requirements. You can download yolov5 from here:

```bash
https://github.com/ultralytics/yolov5
```

Make sure to install all requirements for our project:

```bash
pip install -r requirements.txt
```

## How to run this ALPR (Automatic License Plate Recognition)
### Specify paths
Specify yolov5 paths in `yolo_license_plate/yolo_main.py` as below:

```bash
# Define the path to your YOLOv5 directory
yolov5_path = 'yolov5'  # Adjust this path if needed
image_folder = Path('yolo_license_plate/dataset/images/tutor')  # Folder containing the images

# Define paths for the trained model weights and the image you want to test
weights_path = 'runs/train/license_plate_model4/weights/best.pt'  # Path to your trained model weights
image_path = 'dataset/images/tutor'  # Replace with the path to the image you want to test
confidence_threshold = 0.6 # how confident yolov5 needs to be before marking a license plate
```

Specify cropped images paths in `ocr.py` as below

```bash
# Define the directory paths
image_folder = Path(f'{ROOT}/runs/detect/test_output8/crops/license_plate')  # Folder containing the images
output_eval_folder = Path(f'{ROOT}/results') # Folder where you want to save images with information about their correct text ratio

```

### Run program to detect license plate
In order to run a YOLOv5 to detect license plate and crop ROI (Region Of Interest), run:
`python yolo_license_plate/yolo_main.py`

In order to run OCR to get license plate text from cropped ROI, run:
`python ocr.py`

## Known issues

### Cannot instantiate WindowsPath / PosixPath

If you encounter the following issue:

```
  File "/usr/lib64/python3.12/pathlib.py", line 1434, in __new__
    raise NotImplementedError(
NotImplementedError: cannot instantiate 'WindowsPath' on your system
```

You need to edit yolov5  `detect.py` file and add the following (below example was made for Linux)

```
if(os.uname()[0] == 'Linux'):
    import pathlib
    temp = pathlib.PosixPath
    pathlib.WindowsPath = pathlib.PosixPath
```
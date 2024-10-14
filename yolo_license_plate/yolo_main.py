import os
import cv2  
import pandas as pd
import subprocess
from pathlib import Path

# Define the path to your YOLOv5 directory

yolov5_path = 'yolov5'  # Adjust this path if needed
image_folder = Path('yolo_license_plate/dataset/images/easy')  # Folder containing the images
label_folder = Path('yolo_license_plate/runs/detect/test_output6/labels')  # Folder containing the labels



# Define paths for the trained model weights and the image you want to test
weights_path = 'runs/train/license_plate_model4/weights/best.pt'  # Path to your trained model weights
image_path = 'dataset/images/easy'  # Replace with the path to the image you want to test
confidence_threshold = 0.6
# Output folder for cropped images
output_crop_folder = Path('runs/detect/test_output/crops')
os.makedirs(output_crop_folder, exist_ok=True)

# Construct the command to run YOLOv5 detection
command = [
    "python", f"{yolov5_path}/detect.py",
    "--weights", weights_path,
    "--data", "data.yaml",
    "--source", image_path,
    "--img", "640",  # Use the same image size as during training
    "--conf", "0.25",  # Confidence threshold (adjust as needed)
    "--save-txt",  # Save the detection results in a .txt file
    "--save-conf",  # Save confidence scores with the results
    "--project", "runs/detect",  # Folder to save the detection results
    "--name", "test_output",  # Name for the subfolder where results will be saved
    "--save-crop",
    f"--conf-thres={confidence_threshold}"
]

# Run the YOLOv5 detection command

print(' '.join(command))
subprocess.run(command)

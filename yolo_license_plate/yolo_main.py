import os
import subprocess
import cv2  # OpenCV to crop images
import pandas as pd  # To read YOLOv5 results
from pathlib import Path

# Define the path to your YOLOv5 directory
yolov5_path = 'yolov5'  # Adjust this path if needed

# Change to the YOLOv5 directory
os.chdir(yolov5_path)

# Define paths for the trained model weights and the image you want to test
weights_path = 'C:/Users/bogda/yolo_license_plate/yolov5/runs/train/license_plate_model/weights/best.pt'  # Path to your trained model weights
image_path = 'C:/Users/bogda/licenseplate-recognition/yolo_license_plate/dataset/images/validate'  # Replace with the path to the image you want to test

# Output folder for cropped images
output_crop_folder = 'runs/detect/test_output/crops'
os.makedirs(output_crop_folder, exist_ok=True)

# Construct the command to run YOLOv5 detection
command = [
    "python", "detect.py",
    "--weights", weights_path,
    "--source", image_path,
    "--img", "640",  # Use the same image size as during training
    "--conf", "0.25",  # Confidence threshold (adjust as needed)
    "--save-txt",  # Save the detection results in a .txt file
    "--save-conf",  # Save confidence scores with the results
    "--project", "runs/detect",  # Folder to save the detection results
    "--name", "test_output",  # Name for the subfolder where results will be saved
]

# Run the YOLOv5 detection command
subprocess.run(command)

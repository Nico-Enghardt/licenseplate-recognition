import os
import subprocess

# Define the path to your YOLOv5 directory
yolov5_path = 'yolov5'  # Adjust this path if needed

# Change to the YOLOv5 directory
os.chdir(yolov5_path)

# Define paths for the trained model weights and the image you want to test
weights_path = 'C:/Users/bogda/yolo_license_plate/yolov5/runs/train/license_plate_model/weights/best.pt'  # Path to your trained model weights
image_path = 'C:/Users/bogda/Desktop/6663dps-1.jpg'  # Replace with the path to the image you want to test

# Construct the command as a list
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

# Run the command
subprocess.run(command)

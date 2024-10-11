import os
import subprocess

# Define the path to your YOLOv5 directory
yolov5_path = 'C:/Users/bogda/yolov5'  # Replace with the path to the YOLOv5 folder

# Change to the YOLOv5 directory
os.chdir(yolov5_path)

# Define paths for the dataset and YAML file
data_yaml = 'C:/Users/bogda/yolo_license_plate/data.yaml'  # Absolute or relative path to your YAML file
weights = 'yolov5s.pt'  # Pre-trained weights file

# Number of epochs and image size (adjust based on your dataset and hardware)
epochs = 100
img_size = 640
batch_size = 16

# Construct the command as a list
command = [
    "python", "train.py",
    "--img", str(img_size),
    "--batch", str(batch_size),
    "--epochs", str(epochs),
    "--data", data_yaml,
    "--weights", weights,
    "--cache"
]

# Run the command
subprocess.run(command)

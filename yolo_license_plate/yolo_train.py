import os
import subprocess

# Define the path to your YOLOv5 directory
yolov5_path = 'yolov5'  # Now it is relative since it's in the same folder as the script

# Change to the YOLOv5 directory
os.chdir("yolo_license_plate/yolov5")

# Define paths for the dataset and YAML file
data_yaml = '../data.yaml'  # Relative path to the YAML file since it's one level up
weights = 'runs/train/license_plate_model3/weights/best.pt'  # Pre-trained weights file, which can be automatically downloaded

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
    "--weights", weights,  # Correctly specifying the pre-trained weights file
    "--project", "runs/train",  # Specify the folder to save results
    "--name", "license_plate_model",  # Name for the subfolder where results will be saved
    "--cache"
]

# Run the command
subprocess.run(command)

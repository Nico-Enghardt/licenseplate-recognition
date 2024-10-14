import os
import subprocess
from sys import path
from pathlib import Path



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in path:
    path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Define the path to your YOLOv5 directory
yolov5_path = f'{ROOT}/yolov5'  # Now it is relative since it's in the same folder as the script

# Define paths for the dataset and YAML file
data_yaml = 'data.yaml'  # Relative path to the YAML file since it's one level up
weights = 'runs/train/license_plate_model4/weights/best.pt'  # Pre-trained weights file, which can be automatically downloaded

# Number of epochs and image size (adjust based on your dataset and hardware)
epochs = 100
img_size = 640
batch_size = 16

# Construct the command as a list
command = [
    "python", f"{yolov5_path}/train.py",
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


# once we detected images, we need to crop them and save them to a file

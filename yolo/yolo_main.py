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
yolov5_path = f'{ROOT}/yolov5'  # Adjust this path if needed
image_folder = Path(f'{ROOT}/yolo_license_plate/dataset/images/tutor')  # Folder containing the images

# Define paths for the trained model weights and the image you want to test
weights_path = f'{ROOT}/runs/train/license_plate_model4/weights/best.pt'  # Path to your trained model weights
image_path = f'{ROOT}/dataset/images/tutor'  # Replace with the path to the image you want to test
confidence_threshold = 0.6
# Output folder for cropped images
output_crop_folder = Path('runs/detect/test_output/crops')
os.makedirs(output_crop_folder, exist_ok=True)


# Construct the command to run YOLOv5 detection
command = [
    "python", f"{yolov5_path}/detect.py",
    "--weights", f"{weights_path}",
    "--data", "data.yaml",
    "--source", f"{image_path}",
    "--img", "640",  # Use the same image size as during training
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

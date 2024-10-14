import os
import subprocess

# Define the path to your YOLOv5 directory

yolov5_path = 'yolov5'  # Adjust this path if needed

# Change to the YOLOv5 directory
os.chdir('yolo_license_plate/yolov5')

# Define paths for the trained model weights and the image you want to test
weights_path = 'runs/train/license_plate_model3/weights/best.pt'  # Path to your trained model weights
image_path = '../dataset/images/validate'  # Replace with the path to the image you want to test

# Output folder for cropped images
output_crop_folder = 'runs/detect/test_output/crops'
os.makedirs(output_crop_folder, exist_ok=True)

# Construct the command to run YOLOv5 detection
command = [
    "python", "detect.py",
    "--weights", weights_path,
    "--data", "data.yaml",
    "--source", image_path,
    "--img", "640",  # Use the same image size as during training
    "--conf", "0.25",  # Confidence threshold (adjust as needed)
    "--save-txt",  # Save the detection results in a .txt file
    "--save-conf",  # Save confidence scores with the results
    "--project", "runs/detect",  # Folder to save the detection results
    "--name", "test_output",  # Name for the subfolder where results will be saved
]

# Run the YOLOv5 detection command

print(command)

subprocess.run(command)

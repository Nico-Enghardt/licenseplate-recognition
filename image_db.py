import numpy as np
import cv2
import os

source_folders = ["Frontal", "Lateral"]

def get_images() -> list[(str, np.array)]:
    labels = []
    images = []
    
    for folder in source_folders:
        image_paths = os.listdir(os.path.join("Images", folder))
        
        for image_name in image_paths:

            
            license_plate = image_name[0:7] 
            labels.append(license_plate)
            image_path = os.path.join("Images", folder, image_name)
            image = cv2.imread(image_path) 
            
            if image is not None: 
                images.append((image, image_name))

    return labels, images

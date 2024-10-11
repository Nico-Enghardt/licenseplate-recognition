
import numpy as np
import cv2
import os

source_folders = ["Frontal", "Lateral", "Nico"]

def get_images(max=10, exclude_tags="onb") -> list[(str, np.array)]:
    # Tags to include optionally are 
    # -n: night
    # -o: ocluded
    # -b: motorbike
    
    labels = []
    images = []
    
    for folder in source_folders:
        
        image_paths = os.listdir(os.path.join("Images", folder))
        
        for image_name in image_paths:
            
            # Exptected Label Schema: 1234UAB-on.jpg
            # <numbers><LETTERS>-<tags>.jpg
            
            license_plate = image_name[0:7]
            tags = os.path.basename(image_name)[9:-4]
            
            if any(t in exclude_tags for t in tags):
                
                continue
            
            labels += [license_plate]
            images += [cv2.imread(os.path.join("Images", folder, image_name))]
            
    if len(images) < max:
        max = len(images)       
    
    return labels[:max], images[:max]
            
labels, images = get_images()

print(labels)
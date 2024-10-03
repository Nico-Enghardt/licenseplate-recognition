
import numpy as np
import cv2
import os

source_folders = ["Frontal", "Lateral", "Nico", "Others"]

def get_data(max=10, exclude_tags="onb", source_folders=source_folders) -> iter:
    # Tags to include optionally are 
    # -n: night
    # -o: ocluded
    # -b: motorbike
    
    labels = []
    images = []
    
    for folder in source_folders:
        
        image_paths = os.listdir(os.path.join("Images", folder))
        
        return_count = 0
        
        for image_name in image_paths:
            
            if return_count > max: # Stop function execution
                return
            
            # Exptected Label Schema: 1234UAB-on.jpg
            # <numbers><LETTERS>-<tags>.jpg
            
            license_plate = image_name[0:7]
            tags = os.path.basename(image_name)[9:-4]
            
            if any(t in exclude_tags for t in tags):
                
                continue
            
            label = license_plate
            image = cv2.imread(os.path.join("Images", folder, image_name))
            
            yield label, image
            
            return_count += 1
            
if __name__ == "__main__":
    
    for i, (label, image) in enumerate(get_data()):
        print(i, label)

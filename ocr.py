import cv2  
from pathlib import Path
import numpy as np
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR 


def showPic(img, title="Image", width=800, height=600):
    #img = cv2.resize(img, (width, height))
    cv2.imshow(title, img)
    print(title)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parseLpText(text):
    text = text.replace(" ","")
    for i in range(4, len(text)):
        if i >= 4 and len(text) - i >= 3:
            if text[i - 1].isdigit() and text[i].isalpha():
                nums = text[i - 4:i]
                chars = text[i:i + 3]
                if nums.isdigit() and chars.isalpha():
                    return text[i - 4:i + 3]
    return ''

def getLpText(d):
    maxc = 0
    text = ''
    for key in d:
        if d[key] > maxc:
            text = key
            maxc = d[key]
    return text

def getLPfromFileName(fp):
    fp = fp.split('/')
    

# Define the directory paths
# Validation Set ----
# image_folder = Path('yolo_license_plate/dataset/images/validate')  # Folder containing the images
# label_folder = Path('yolo_license_plate/yolov5/runs/detect/test_output6/labels')  # Folder containing the labels

# Easy Set (Tutor-given images)
image_folder = Path('yolo_license_plate/runs/detect/test_output8/crops/license_plate')  # Folder containing the images
output_eval_folder = Path('results')

# Ensure the output directory for cropped images exists
output_eval_folder.mkdir(parents=True, exist_ok=True)

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

scores = []
evals = []

# Iterate over all images in the image folder
for t, image_path in enumerate(image_folder.glob('*.jp*g')):  # Adjust file extension as needed (e.g., .png)
    # Generate the corresponding label file path

    print(f'processing Img. Nr {t}', image_path)

    # Check if the label file exists
    if image_path.is_file(): #TODO UPDATE THIS

        license_plate = cv2.imread(image_path) 
        lpText = ""
        try:
            lpText = "".join(k[1][0] for k in paddle_ocr.ocr(license_plate)[0])
        except Exception as err: 
            print("Tesseract error:", err)
        recognized_text = parseLpText(lpText)
        print('Recognized text: ',recognized_text)

        correctLP = image_path.stem[:7]
        print(correctLP)
        
        if recognized_text == correctLP:
            print('correct found!!!!!')
            scores += [100]
            evals += ["perfect"]

            eval_image_path = f"{output_eval_folder}/{image_path.stem}_correct_pad.jpg"
            print(eval_image_path)
            cv2.imwrite(str(eval_image_path), license_plate)
    
        elif recognized_text:
            score = fuzz.ratio(recognized_text, correctLP)
            scores += [score]
            evals += ["partial match"]
            print(f"Could achieve score of {score}")
            
            eval_image_path = f"{output_eval_folder}/{image_path.stem}_{score}%_pad.jpg"
            cv2.imwrite(str(eval_image_path), license_plate)
        
        else:
            evals += ["ocr_failure"]
            scores += [0]
            eval_image_path = f"{output_eval_folder}/{image_path.stem}_nodetection.jpg"
            cv2.imwrite(str(eval_image_path), license_plate)
                
    else: 
        scores += [0]
        evals += ["no plate"]
                
with open("results.txt", "w") as file:
    file.write(f'{scores}\n')
    
    file.write(f'{evals}\n')
    file.write(f'Average score: {np.mean(scores)}\n')
    file.write(f'Median score: {np.median(scores)}\n')
    file.write(f'correct count: {evals.count("perfect")}/{len(evals)}')
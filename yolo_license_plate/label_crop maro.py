import cv2  
import pandas as pd
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
    
def median_brightness(img, percentile = 50):
    pixels = img.flatten()
    sorted = np.sort(pixels)
    
    return sorted[int(np.floor(percentile/100*len(sorted)))]

def processLp(img, thresh_perc=50):

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gImg,(9,9),2)
    
    threshold = median_brightness(img, percentile=thresh_perc)
    
    threshInv = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1] # we may want to try with otsu later?

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    threshInv = cv2.erode(threshInv,kernel,iterations=1)
    #showPic(threshInv)
    threshInv = cv2.dilate(threshInv,kernel,iterations=1)
    # showPic(threshInv)

    return threshInv

def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)
    return pts[ind]

def getContours(img, orig, ):  # Change - pass the original image too
    image = orig.copy()
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    warped = None  # Stores the warped license plate image
    if index is not None: # Draw the biggest contour on the image
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)

        src = np.squeeze(biggest).astype(np.float32) # Source points
        height = image.shape[0]
        width = image.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        biggest = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_LINEAR)

    return biggest, imgContour, warped  # Change - also return drawn image

def straighten_lp(image):
    kernel = np.ones((15,10))
    first_erode_kernel = np.ones((2,2))
    # image = cv2.imread(image)


    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),2)
    imgCanny = cv2.Canny(imgBlur,150,200)
    # imgErode1 = cv2.erode(imgCanny, first_erode_kernel, iterations=1)
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    biggest, imgContour, warped = getContours(imgThres, image)  # Change

    titles = ['Original', 'Blur', 'Canny', 'Dilate', 'Threshold', 'Contours', 'Warped']  # Change - also show warped image
    images = [image[...,::-1],  imgBlur, imgCanny, imgDial, imgThres, imgContour, warped]  # Change

    return warped, titles, images

# Define the directory paths
image_folder = Path('yolo_license_plate/dataset/images/validate')  # Folder containing the images
label_folder = Path('yolo_license_plate/yolov5/runs/detect/test_output9/labels')  # Folder containing the labels
output_eval_folder = Path('yolo_license_plate/results')

# Ensure the output directory for cropped images exists
output_eval_folder.mkdir(parents=True, exist_ok=True)

# Define a scaling factor to reduce the size of the crop (e.g., 0.8 for 80% size)
crop_scale = 1.3

alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
options += " --psm {}".format(11)

# Set EasyOcr options
# reader = easyocr.Reader(['en']) # specify the language

paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')

scores = []
evals = []

# Iterate over all images in the image folder
for image_path in image_folder.glob('*.jpg'):  # Adjust file extension as needed (e.g., .png)
    # Generate the corresponding label file path
    label_path = label_folder / f"{image_path.stem}.txt"  # Match the label to the image by name

    print('processing: ', image_path)

    # Check if the label file exists
    if label_path.is_file():
        # Read the image using OpenCV
        image = cv2.imread(str(image_path))
        
        # Read the YOLOv5 labels (bounding boxes) from the .txt file
        cols = ['class', 'x_center', 'y_center', 'width', 'height', 'confidence']
        detections = pd.read_csv(label_path, sep=' ', names=cols)

        # Get image dimensions
        h, w, _ = image.shape
        
        # Loop over each detection and crop the area
        for index, row in detections.iterrows():
            lpDict = {}
            # Convert YOLO format to pixel coordinates
            x_center = int(row['x_center'] * w)
            y_center = int(row['y_center'] * h)
            box_width = int(row['width'] * w * crop_scale)  # Scale down the width
            box_height = int(row['height'] * h * crop_scale)  # Scale down the height
            
            # Calculate the top-left corner of the bounding box
            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = x_min + box_width
            y_max = y_min + box_height
            
            # Ensure the bounding box coordinates are within image dimensions
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)


            license_plate = image[y_min:y_max, x_min:x_max]

            # crop_image_path = output_eval_folder / f"{image_path.stem}_crop_{index}.jpg"
            # cv2.imwrite(str(crop_image_path), license_plate)
                
            lpText = ""
            try:
                lpText = "".join(k[1][0] for k in paddle_ocr.ocr(license_plate)[0])
                #lpText =  "".join([text for _,text,_ in reader.readtext(proccessed_lp)])
                #lpText = pytesseract.image_to_string(proccessed_lp, config=options)
            
            except Exception as err: 
                print("Tesseract error:", err)
            parsed = parseLpText(lpText)
            if parsed:
                # showPic(license_plateg)
                lpDict[parsed] = lpDict.get(parsed, 0)+1

        best_hit =  getLpText(lpDict)
        print('best hit: ',best_hit)
        # correctLP = imgfile.split('/')[-1][:-4]
        correctLP = image_path.stem
        
        if best_hit == correctLP:
            print('correct found!!!!!')
            scores += [100]
            evals += ["perfect"]

            eval_image_path = output_eval_folder / f"../eval/{image_path.stem}_correct_pad_{index}.jpg"
            cv2.imwrite(str(eval_image_path), license_plate)
    
        elif best_hit:
            score = fuzz.ratio(best_hit, correctLP)
            scores += [score]
            evals += ["partial match"]
            print(f"Could achieve score of {score}")
            
            eval_image_path = output_eval_folder / f"../eval/{image_path.stem}_{score}%_pad_{index}.jpg"
            cv2.imwrite(str(eval_image_path), license_plate)
        
        else:
            evals += ["ocr_failure"]
            scores += [0]
            eval_image_path = output_eval_folder / f"../eval/{image_path.stem}_nodetection_{index}.jpg"
            cv2.imwrite(str(eval_image_path), license_plate)
                
    else: 
        scores += [0]
        evals += ["no plate"]
                
with open("results.txt", "w") as file:
    file.write(f'{scores}')
    
    file.write(f'{evals}')
    file.write(f'Average score: {np.mean(scores)}')
    file.write(f'Median score: {np.median(scores)}')
    file.write(f'correct count: {np.sum(np.where(scores==100))}')
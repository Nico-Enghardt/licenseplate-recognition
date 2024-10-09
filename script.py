import argparse
import cv2
import numpy as np
import pytesseract
import os
from collections import Counter
from image_db import *

correct_plates = 0
total_images = 0

def showPic(img, title="Image", width=800, height=600):
    img = cv2.resize(img, (width, height))
    cv2.imshow(title, img)
    print(title)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def parseLpText(text):
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

def processImg(imgfile, filename, template):
    global correct_plates, total_images

    lpDict = {}
    correct_plate = filename[:7]  # Assuming the first 7 chars are the correct plate number
    imS = imgfile.copy()

    gray_img = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    

    result = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5  
    locations = np.where(result >= threshold)

    for pt in zip(*locations[::-1]):
        x, y = pt
        h, w = template.shape[:2]
        cv2.rectangle(imS, (x, y), (x + w, y + h), (0, 255, 0), 2)
        license_plate = imS[y:y + h, x:x + w]
        #showPic(license_plate, title="Image", width=800, height=600)
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = f"-c tessedit_char_whitelist={alphanumeric} --psm 7"
        lpText = pytesseract.image_to_string(license_plate, config=options)
        
        if lpText and len(lpText) >= 7:
            parsed = parseLpText(lpText)
            if parsed:
                lpDict[parsed] = lpDict.get(parsed, 0) + 1

    recognized_plate = getLpText(lpDict)
    print('best hit: ', recognized_plate)
    
    total_images += 1
    if recognized_plate == correct_plate:
        correct_plates += 1


def display_accuracy():
    if total_images == 0:
        print("No images processed.")
    else:
        accuracy = (correct_plates / total_images) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct_plates}/{total_images})")

template = cv2.imread('template.jpg')  

lab, img = get_images()

for image, filename in img:
    processImg(image, filename, template)

display_accuracy()
cv2.destroyAllWindows()

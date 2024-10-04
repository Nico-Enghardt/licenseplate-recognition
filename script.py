import argparse
import imutils
import cv2
import pytesseract
import os
from image_db import *

correct_plates = 0
total_images = 0

def processAllInFolder(dir):
    for file in os.listdir(dir):
        f = os.path.join(dir, file)
        print(f)
        if os.path.isfile(f):
            processImg(f)

def saveImage(name, file, img, directory='test'):
    filename = file.split('/')[-1]
    filename = directory + '/' + filename[:-4] + name + '.jpg'
    cv2.imwrite(filename, img)

def showPic(img, title="Image"):
    img = cv2.resize(img, (1920, 1080))
    cv2.imshow("Image", img)
    print(title)
    cv2.waitKey(0)

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

def processLp(img):
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    tophat = cv2.morphologyEx(gImg, cv2.MORPH_TOPHAT, rectkernel)
    blackhat = cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectkernel)
    threshInv = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY_INV)[1]
    threshInv = cv2.erode(threshInv, kernel, iterations=2)
    threshInv = cv2.dilate(threshInv, kernel, iterations=2)
    return threshInv

def processImg(imgfile,filename):
    global correct_plates, total_images

    lpDict = {}
    correct_plate = filename[:7]  # Assuming the first 7 chars are the correct plate number
    imS = imgfile.copy()

    gImg = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closingkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 15))
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 30))
    
    blurred = cv2.GaussianBlur(gImg, (3, 3), 0)
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, rectKernel)
    blackhat = cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectKernel)
    
    threshInv = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)[1]
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(11)

    for i in range(5):
        for j in range(5):
            eroded = cv2.erode(threshInv, kernel, iterations=1 + i)
            dilated = cv2.dilate(eroded, kernel, iterations=1 + j)
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closingkernel, iterations=5)
            contours = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            min_width = 300
            min_height = 50
            test = imS.copy()

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 2.5 < aspect_ratio < 6.0 and w >= min_width and h >= min_height:
                    license_plate = imS[y:y + h + 10, x:x + w + 10]
                    license_plateg = processLp(license_plate)
                    lpText = pytesseract.image_to_string(license_plateg, config=options)
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

# MAIN FUNC BELOW

dir = 'Images/Lateral'
lab, img = get_images()

for image,filename in img:
    processImg(image,filename)

display_accuracy()

cv2.destroyAllWindows()
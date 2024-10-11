import argparse
import imutils
import cv2
import pytesseract
import os
import time
import numpy as np
import json

from image_db import get_data

erosion_iterations = 5


def processAllInFolder(dir):
    for file in os.listdir(dir):
        f = os.path.join(dir, file)
        print(f)
        if os.path.isfile(f):
            processImg(f)

def saveImage(name,label,img,directory='test/bboxes', dict=None):
    if dict:
        with open(directory + '/'+label + name + ".json", "w") as file: 
            json.dump(dict, file)
    filename = directory + '/'+label + name + '.jpg'
    cv2.imwrite(filename,img)


def showPic(img,title="Image"):
    img = cv2.resize(img, (1920,1080))
    cv2.imshow("Image",img)
    #print(title)
    cv2.waitKey(0)



def parseLpText(text):
    # find index where there is a number and letter, then get 4 letters before and 3 chars after,
    # because sometimes ocr recognises "E" sign at the beggining as different characters
    text = text.replace(" ","")
    for i in range(4,len(text)):
            if i>=4 and len(text)-i >= 3:
                if text[i-1].isdigit() and text[i].isalpha():
                    nums = text[i-4:i]
                    chars = text[i:i+3]
                    if nums.isdigit() and chars.isalpha():
                        print('license: ', text[i-4:i+3])
                        return text[i-4:i+3], nums, chars
    return '', '', ''
         

def getMax(d):
    maxc = 0
    text = ''
    for key in d:
        if d[key] > maxc:
            text = key
            maxc = d[key]
            
    return text, maxc

def getLpText(d):
    
    best_num, maxc_num = getMax(d["nums"])
    best_char, maxc_char = getMax(d["chars"])

    return best_num + best_char, min(maxc_num, maxc_char)


def processLp(img, thresh=60):

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    tophat = cv2.morphologyEx(gImg, cv2.MORPH_TOPHAT, rectkernel)
    # showPic(blurred)  

    #showPic(tophat,'tophat')
    
    blackhat= cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectkernel)
    #showPic(blackhat,'blackhat')
    
    threshInv = cv2.threshold(blackhat, thresh, 255,
    cv2.THRESH_BINARY_INV)[1]

    #edges = cv2.Canny(blurred,100,150)
    #showPic(threshInv)

    threshInv = cv2.erode(threshInv,kernel,iterations=2)
    #showPic(threshInv)
    threshInv = cv2.dilate(threshInv,kernel,iterations=2)
    #showPic(threshInv)
    return threshInv


def processImg(imgfile, image):
    start = time.time()
    lpDict = {"nums": {}, "chars": {}}
    img = image
    imS = img.copy()


    #convert to grayscale
    gImg = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)


    #do opening to get rid of noise?
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    closingkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,15))

    #blurred = cv2.GaussianBlur(gImg, (5,5),0)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,30))
    
    blurred = cv2.GaussianBlur(gImg, (3,3),0)
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, rectKernel)
    # showPic(blurred)  

    # showPic(tophat,'tophat')
    
    blackhat= cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectKernel)
    #showPic(blackhat,'blackhat')
    
    threshInv = cv2.threshold(blackhat, 100, 255,
	cv2.THRESH_BINARY )[1]

    #edges = cv2.Canny(blurred,100,150)
    #showPic(threshInv)
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
		# set the PSM mode
    options += " --psm {}".format(7)
    
    test = imS.copy()
    
    found_lp = 0

    for i in range (erosion_iterations):
        for j in range(erosion_iterations):
            eroded = cv2.erode(threshInv,kernel,iterations=1+i) #changing those values drasticaly changes output, therefore we can add another iteration if we did not find any good areas
            dilated = cv2.dilate(eroded,kernel,iterations=1+j)
            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closingkernel,iterations=5)

            contours = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            min_area = 500*100 # @Todo Make relative to image size
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                cv2.rectangle(test, (x,y), (x+w,y+h), (255,100,100),1)
                
                
                if .7 < aspect_ratio < 6.0 and w*h > min_area :  # Adjust this range based on the license plate shape
                    cv2.rectangle(test, (x,y), (x+w,y+h), (0,255,0),3)
                    license_plate_contour = cnt

                    x, y, w, h = cv2.boundingRect(license_plate_contour)

                    license_plate = imS[y:y+h+30, x:x+w+30]
                    
                    #showPic(license_plate)
                    for k in [75]:
                        license_plateg = processLp(license_plate, thresh=k)
                        #showPic(license_plateg)
                                
                        lpText = pytesseract.image_to_string(license_plateg, config=options)

                        if lpText and len(lpText) >= 3:
                            found_lp += 1
                            saveImage(f"lplate-{found_lp}" , imgfile, license_plateg, directory="test/licenseplates")

                            _, nums, chars = parseLpText(lpText)
                            if nums:
                                lpDict["nums"][nums] = lpDict["nums"].get(nums, 0)+1
                            if chars:
                                lpDict["chars"][chars] = lpDict["chars"].get(chars, 0)+1
        
            if getLpText (lpDict)[1] > 5:
                break
            
        if getLpText(lpDict)[1] > 5:
            break
                            
    print(lpDict)
                                
    if found_lp == 0:
        print(imgfile)
        saveImage(f"lplate-notfound" , imgfile, imS, directory="test/notfound")
        saveImage(f"lplate-notfound" , imgfile, test, directory="test/notfound")
    prediction = getLpText(lpDict)[0]             
        
    if prediction != imgfile and len(prediction) == 7:
        saveImage(f"wrong-pred", imgfile, test, directory="test/wrong-pred", dict=lpDict)
        
    print('best hit: ', prediction)
    print(imgfile)
    saveImage('bin',imgfile,threshInv)
    saveImage('bbox',imgfile,test, directory="test/bbox")
    
    end = time.time()
    
    return prediction, end-start, prediction==imgfile

durations = []
corrects = []
bad_labels = []

for i, (label, img) in enumerate(get_data(max=30,exclude_tags="no")):  
    print(i)
    
    # label="9892JFR"
    # img = cv2.imread("Images/Frontal/"+ label+ ".jpg")
    
    # if label not in ["0907JRF",
    #              "3660CRT",
    #              "1498JBZ",
    #              "5789JHB",
    #              "8727JTC",
    #              "6401JBX",
    #              "1556GMZ",
    #              "7153JWD",
    #              "8727JTC",
    #              "3340JMF",
    #              "9247CZG"]:
    #     continue
    

    pred, duration, correct = processImg(label, img)

    print("Duration:", duration)
    if correct:
        durations += [duration]
        print("Prediction correct!")

    else: bad_labels += [(label, pred)]
    corrects += [correct]
    

print(f"{sum(corrects)} of {len(corrects)} are correct, {sum(corrects)/len(corrects)*100.:0f} %")
print(f"Average time: {np.average(duration)}")
print("Incorrectly detected labels:", bad_labels)

cv2.destroyAllWindows()

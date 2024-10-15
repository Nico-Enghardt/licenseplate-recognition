import argparse
import imutils
import cv2
import pytesseract
import os
import numpy as np
from matplotlib import pyplot as plt
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR 

correct_count = 0
processed_count = 0




scores = []
evals = []
paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log = False)



def processImg(imgfile):
    global correct_count
    global processed_count
    global scores
    global evals
    global paddle_ocr

    dialetkernel_arr = [ (50,20),(30,30), (20,20), (7,7)]

    hatkernel_arr = [ (300,60) ,(150,60),  (150,30)]
    threshold_arr = [140,130,120,100]
    lpDict = {}
    img = cv2.imread(imgfile)
    imS = img.copy()
    #imS = cv2.resize(img, (1920,1080))
    below_5_matches = True
    for dilate_first in range (1):
        
        for dialetkernel in dialetkernel_arr:
            for hatkernel in hatkernel_arr:
                for threshold in threshold_arr: 
                    if below_5_matches:
                        gImg = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
                        closingkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,10))
                        dilatekernel = cv2.getStructuringElement(cv2.MORPH_RECT, dialetkernel)

                        blurred = cv2.GaussianBlur(gImg, (9,9),2)

                        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, hatkernel ) ## changing this one looks like chanigng a lot

                        tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, rectKernel)

                        blackhat= cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectKernel)
        
                        threshInv = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY )[1]
                        edges = cv2.Canny(threshInv,150,200)
                        for i in range (2):
                            for j in range(2):
                                if dilate_first:
                                    eroded = cv2.erode(edges,kernel,iterations=2+i) #changing those values drasticaly changes output, therefore we can add another iteration if we did not find any good areas
                                    dilated = cv2.dilate(eroded,dilatekernel,iterations=2+j)

                                else:
                                    dilated = cv2.dilate(threshInv,dilatekernel,iterations=2+j)
                                    eroded = cv2.erode(dilated,kernel,iterations=2+i) #changing those values drasticaly changes output, therefore we can add another iteration if we did not find any good areas

                                closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closingkernel,iterations=1)
                                saveImage('closing',imgfile,closing)


                                edges = cv2.Canny(closing,50,290)
                                contours = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                                contours = imutils.grab_contours(contours)
                                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                                min_width = 400
                                min_height = 50
                                max_height = 400
                                max_width = 1000
                                test = imS.copy()
                                for cnt in contours:
                                    x, y, w, h = cv2.boundingRect(cnt)
                                    aspect_ratio = w / float(h)
                                    cv2.rectangle(test, (x,y), (x+w,y+h), (0,255,0),3)
                                    if 3.0 < aspect_ratio < 6.0 and w >= min_width and h >= min_height and w <=max_width and h <= max_height :  # Adjust this range based on the license plate shape
                                        license_plate_contour = cnt

                                        x, y, w, h = cv2.boundingRect(license_plate_contour)
                                        tmp = 50
                                        if x >= tmp and y >= tmp:

                                            license_plate = imS[y-tmp:y+h+tmp, x-tmp:x+w+tmp]

                                            cv2.rectangle(img, (x-tmp,y-tmp), (x+w+tmp,y+h+tmp), (0,255,0),3)
                                        else:
                                            license_plate = imS[y:y+h+tmp, x:x+w+tmp]
                                            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)

                                        lpText = ""
                                        try:
                                            lpText = "".join(k[1][0] for k in paddle_ocr.ocr(license_plate)[0])
                                            print('Recognized text: ',lpText)

                                            if lpText and len(lpText) >= 7:
                                                # print(lpText)

                                                parsed = parseLpText(lpText)
                                                if parsed:
                                                    lpDict[parsed] = lpDict.get(parsed, 0)+1
                                                    if lpDict.get(parsed) >= 5:
                                                        below_5_matches = False
                                        except Exception as err: 
                                            continue

    processed_count +=1
    best_hit =  getLpText(lpDict)
    print('best hit: ',best_hit)
    correctLP = imgfile.split('/')[-1][:7]
    if best_hit == correctLP:
        correct_count += 1
        print('correct found!!!!!')
        scores += [100]
        evals += ["perfect"]

    elif best_hit:
        score = fuzz.ratio(parsed, correctLP)
        scores += [score]
        evals += ["partial match"]
        print(f"Could achieve score of {score}")

    else:
        evals += ["ocr_failure"]
        scores += [0]

    print(imgfile)


def processAllInFolder(dir):
    for file in os.listdir(dir):
        f = os.path.join(dir, file)
        print(f)
        if os.path.isfile(f):
            processImg(f)

def saveImage(name,file,img,directory='test'):
    filename = file.split('/')[-1]
    filename = directory + '/'+filename[:-4] + name + '.jpg'
    cv2.imwrite(filename,img)


def showPic(img,title="Image"):
    img = cv2.resize(img, (1920,1080))
    cv2.imshow("Image",img)
    print(title)
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
                        return text[i-4:i+3]
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

def processLp(img):

    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    dilatekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6,6))
    tophat = cv2.morphologyEx(gImg, cv2.MORPH_TOPHAT, rectkernel)
    # showPic(blurred)  

    # showPic(tophat,'tophat')
    
    blackhat= cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectkernel)
    # showPic(blackhat,'blackhat')

    
    threshInv = cv2.threshold(blackhat, 100, 255,
	cv2.THRESH_BINARY_INV)[1]

    #edges = cv2.Canny(blurred,100,150)
    # showPic(threshInv)
    


    threshInv = cv2.erode(threshInv,kernel,iterations=2)
    # showPic(threshInv)
    threshInv = cv2.dilate(threshInv,dilatekernel,iterations=2)
    # showPic(threshInv)

    return threshInv

def show_all_images(titles, images):
    for i in range(5):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])

    plt.subplot(3, 3, 6)
    plt.imshow(images[-2])
    plt.title(titles[-2])

    plt.subplot(3, 3, 8)
    plt.imshow(images[-1])
    plt.title(titles[-1])

    plt.show()



#ap = argparse.ArgumentParser()
#ap.add_argument('-i','--image', required=True,
#                help="path to input image")
#args = vars(ap.parse_args())

## MAIN FUNC BELOW


dir = 'Images/tutor'

processAllInFolder(dir)

print('processed: ',processed_count,"/  correct: ",correct_count)
print(f'Average score: {np.mean(scores)}\n')
print(f'Median score: {np.median(scores)}\n')
print(f'correct count: {evals.count("perfect")}/{len(evals)}')



cv2.destroyAllWindows()







import argparse
import imutils
import cv2
import pytesseract
import os
import numpy as np

correct_count = 0
processed_count = 0

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
    tophat = cv2.morphologyEx(gImg, cv2.MORPH_TOPHAT, rectkernel)
    # showPic(blurred)  

    # showPic(tophat,'tophat')
    
    blackhat= cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectkernel)
    # showPic(blackhat,'blackhat')

    
    threshInv = cv2.threshold(blackhat, 100, 255,
	cv2.THRESH_BINARY_INV)[1]

    #edges = cv2.Canny(blurred,100,150)
    # showPic(threshInv)
    


    threshInv = cv2.erode(threshInv,kernel,iterations=1)
    # showPic(threshInv)
    threshInv = cv2.dilate(threshInv,kernel,iterations=1)
    # showPic(threshInv)

    return threshInv

def processLpV2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours to find potential license plate candidates
    license_plate_contour = None
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # A license plate is likely to have 4 points (rectangular shape)
        if len(approx) == 4:
            # Check for the aspect ratio to ensure it is likely a license plate
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6:  # Common aspect ratio range for license plates
                license_plate_contour = approx
       # Rearrange contour points for perspective transform
                pts = license_plate_contour.reshape(4, 2)
    
                # Get a consistent order of the points (top-left, top-right, bottom-right, bottom-left)
                rect = np.zeros((4, 2), dtype="float32")

                # Find the top-left and bottom-right points based on the sum of the coordinates
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]  # top-left
                rect[2] = pts[np.argmax(s)]  # bottom-right

                # Find the top-right and bottom-left based on the difference
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]  # top-right
                rect[3] = pts[np.argmax(diff)]  # bottom-left

                # Define the dimensions of the new perspective (a rectangle)
                width_a = np.linalg.norm(rect[2] - rect[3])
                width_b = np.linalg.norm(rect[1] - rect[0])
                max_width = max(int(width_a), int(width_b))

                height_a = np.linalg.norm(rect[1] - rect[2])
                height_b = np.linalg.norm(rect[0] - rect[3])
                max_height = max(int(height_a), int(height_b))

                # The destination points for the perspective transform (a straight rectangle)
                dst = np.array([
                    [0, 0],
                    [max_width - 1, 0],
                    [max_width - 1, max_height - 1],
                    [0, max_height - 1]
                ], dtype="float32")

                # Apply the perspective transformation matrix
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (max_width, max_height))

                # Show the corrected license plate
                cv2.imshow('Warped License Plate', warped)
                cv2.waitKey(0)



def processImg(imgfile):
    global correct_count
    global processed_count

    all_rects_count = 0 # that meet the size criteria
    all_possible_lp_rects_count = 0 # that after text recognition gave at least 7 chars



    lpDict = {}
    img = cv2.imread(imgfile)
    imS = img.copy()
    #imS = cv2.resize(img, (1920,1080))


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
    # showPic(blackhat,'blackhat')
    
    threshInv = cv2.threshold(blackhat, 100, 255,
	cv2.THRESH_BINARY )[1]

    #edges = cv2.Canny(blurred,100,150)
    # showPic(threshInv)
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
		# set the PSM mode
    options += " --psm {}".format(7)

    for i in range (5):
        for j in range(5):
            eroded = cv2.erode(threshInv,kernel,iterations=1+i) #changing those values drasticaly changes output, therefore we can add another iteration if we did not find any good areas
            # showPic(eroded,'eroded')   
            dilated = cv2.dilate(eroded,kernel,iterations=1+j)
            # showPic(dilated,'dilated')


            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closingkernel,iterations=5)
            # showPic(closing,'closing')
            saveImage('closing',imgfile,closing)


            edges = cv2.Canny(closing,50,290)
            # showPic(edges)
            
            saveImage('edges',imgfile,edges)
            contours = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)
            # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            min_width = 300
            min_height = 50
            # print(contours)
            test = imS.copy()
            for cnt in contours:
                # approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                # if len(approx) == 4:  # Check if the contour has 4 vertices (quadrilateral)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                cv2.rectangle(test, (x,y), (x+w,y+h), (0,255,0),3)
                if 2.5 < aspect_ratio < 6.0 and w >= min_width and h >= min_height:  # Adjust this range based on the license plate shape
                    all_rects_count +=1
                    license_plate_contour = cnt

                    x, y, w, h = cv2.boundingRect(license_plate_contour)
                    # print(w,h)
                    # license_plate = imS[y:y+h+10, x:x+w+10]
                    # license_plateg = processLp(license_plate)
                    license_plate = gImg[y:y+h+10, x:x+w+10]
                    # print("iteration ",i,j)
                    license_plateg = cv2.threshold(license_plate, 100, 255,cv2.THRESH_BINARY )[1]
                    # showPic(license_plateg)
                    # print("shape: ",license_plateg.shape)
                    lpText = pytesseract.image_to_string(license_plateg, config=options)
                    if lpText and len(lpText) >= 7:
                        all_possible_lp_rects_count += 1
                        print(lpText)
                        # cv2.imshow('test',license_plate)
                        cv2.waitKey(0)

                        # cv2.imshow('test',license_plateg)
                        cv2.waitKey(0)
                        parsed = parseLpText(lpText)
                        if parsed:

                            # showPic(license_plateg)
                            lpDict[parsed] = lpDict.get(parsed, 0)+1
                    ## TODO: Once we hace the rectangle of an image, we need to do a Character recognision, to check if there are 4 nums and 3 chars
                    ## Also, we can check if there is a small tall rectangle
                    # cv2.rectangle(imS, (x,y), (x+w,y+h), (0,255,0),3)
                    # cv2.imshow("Image",imS)
                    # cv2.waitKey(0)
                    # cv2.imshow("Image",edges)
                    # cv2.waitKey(0)
                    #break
    processed_count +=1
    best_hit =  getLpText(lpDict)
    print('best hit: ',best_hit)
    print("Processed ",all_rects_count," rectangles.")
    print("Found ", all_possible_lp_rects_count, "rectangles that had at least 7 characters")
    correctLP = imgfile.split('/')[-1][:-4]
    
    if best_hit == correctLP:
        print('correct found!!!!!')
        correct_count += 1
    print(imgfile)
    saveImage('filtered',imgfile,imS)
    saveImage('test',imgfile,test)

#ap = argparse.ArgumentParser()
#ap.add_argument('-i','--image', required=True,
#                help="path to input image")
#args = vars(ap.parse_args())

## MAIN FUNC BELOW


dir = 'Images/Frontal'

# processAllInFolder(dir)
# processImg('Images/Lateral/3044JMB.jpg')
oneImg ='3587DCXlp.jpg' 
img = cv2.imread(oneImg)
processLpV2(img)
print('processed: ',processed_count,"/  correct: ",correct_count)
# print("Ratio: ",processed_count / correct_count)



#cv2.imshow("Image",gImg)
#cv2.waitKey(0)
#cv2.imshow("Image",blurred)
#cv2.waitKey(0)
        #cv2.imshow("Image",edges)
        #cv2.waitKey(0)
        #cv2.imshow("Image",imS)
        #cv2.waitKey(0)


cv2.destroyAllWindows()







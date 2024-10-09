import argparse
import imutils
import cv2
import pytesseract
import os
import numpy as np
from matplotlib import pyplot as plt

correct_count = 0
processed_count = 0

wrong_lp_arr = []

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
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)

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

# Change - Also show contour drawn image + warped image

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
    closingkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,10))
    dilatekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))

    blurred = cv2.GaussianBlur(gImg, (9,9),2)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150,30)) ## changing this one looks like chanigng a lot
    
    # blurred = cv2.GaussianBlur(gImg, (3,3),0)
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, rectKernel)
    # showPic(blurred)  

    showPic(tophat,'tophat')
    
    blackhat= cv2.morphologyEx(tophat, cv2.MORPH_BLACKHAT, rectKernel)
    showPic(blackhat,'blackhat')
    
    threshInv = cv2.threshold(blackhat, 130, 255, cv2.THRESH_BINARY )[1]
    # threshInv = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                            cv2.THRESH_BINARY, 11, 2)

    edges = cv2.Canny(threshInv,150,200)
    showPic(threshInv)
    showPic(edges)
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
		# set the PSM mode
    options += " --psm {}".format(7)

    for i in range (3):
        for j in range(3):
            # eroded = cv2.erode(edges,kernel,iterations=1+i) #changing those values drasticaly changes output, therefore we can add another iteration if we did not find any good areas
            # showPic(eroded,'eroded')   
            # dilated = cv2.dilate(eroded,dilatekernel,iterations=1+j)
            # showPic(dilated,'dilated')

            dilated = cv2.dilate(threshInv,dilatekernel,iterations=1+j)
            # showPic(dilated,'dilated')
            eroded = cv2.erode(dilated,kernel,iterations=1+i) #changing those values drasticaly changes output, therefore we can add another iteration if we did not find any good areas
            # showPic(eroded,'eroded')   

            closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closingkernel,iterations=1)
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
            min_width = 400
            min_height = 50
            max_height = 400
            max_width = 1000
            # print(contours)
            test = imS.copy()
            for cnt in contours:
                # approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                # if len(approx) == 4:  # Check if the contour has 4 vertices (quadrilateral)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                cv2.rectangle(test, (x,y), (x+w,y+h), (0,255,0),3)
                if 2.5 < aspect_ratio < 6.0 and w >= min_width and h >= min_height and w <=max_width and h <= max_height :  # Adjust this range based on the license plate shape
                    all_rects_count +=1
                    license_plate_contour = cnt

                    x, y, w, h = cv2.boundingRect(license_plate_contour)
                    # print(w,h)
                    # license_plate = imS[y:y+h+10, x:x+w+10]
                    # license_plateg = processLp(license_plate)
                    tmp = 50
                    if x >= tmp and y >= tmp:

                        license_plate = imS[y-tmp:y+h+tmp, x-tmp:x+w+tmp]

                        cv2.rectangle(img, (x-tmp,y-tmp), (x+w+tmp,y+h+tmp), (0,255,0),3)
                    else:
                        license_plate = imS[y:y+h+tmp, x:x+w+tmp]
                        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),3)
                    warped, titles, images = straighten_lp(license_plate)

                    if warped is not None:
                        warped = processLp(warped)
                        try:
                            lpText = pytesseract.image_to_string(warped, config=options)
                        except:
                            print("Tesseract error")
                        if lpText and len(lpText) >= 7:
                            all_possible_lp_rects_count += 1
                            print(lpText)
                            cv2.imshow('test',warped)
                            cv2.waitKey(0)

                            # cv2.imshow('test',license_plateg)
                            cv2.waitKey(0)
                            parsed = parseLpText(lpText)
                            if parsed:

                                # showPic(license_plateg)
                                lpDict[parsed] = lpDict.get(parsed, 0)+1
                        # show_all_images(titles,images)
                    # print("iteration ",i,j)
                    # convert it to gray scale
                    license_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                    # blurr a bit

                    license_plate = cv2.GaussianBlur(license_plate,(7,7),2)
                    license_plateg = cv2.threshold(license_plate, 100, 255,cv2.THRESH_BINARY )[1]

                    # showPic(license_plateg)
                    
                    # showPic(license_plateg)
                    # print("shape: ",license_plateg.shape)
                    try:
                        lpText = pytesseract.image_to_string(license_plateg, config=options)
                    except:
                        print("Tesseract error")
                    if lpText and len(lpText) >= 7:
                        all_possible_lp_rects_count += 1
                        print(lpText)
                        # cv2.imshow('test',license_plate)
                        # cv2.waitKey(0)

                        cv2.imshow('test',license_plateg)
                        cv2.waitKey(0)
                        parsed = parseLpText(lpText)
                        if parsed:

                            # showPic(license_plateg)
                            lpDict[parsed] = lpDict.get(parsed, 0)+1
                    ## TODO: Once we hace the rectangle of an image, we need to do a Character recognision, to check if there are 4 nums and 3 chars
                    ## Also, we can check if there is a small tall rectangle
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
    else:
        wrong_lp_arr.append(correctLP)
    print(imgfile)
    saveImage('filtered',imgfile,img)
    saveImage('test',imgfile,test)

#ap = argparse.ArgumentParser()
#ap.add_argument('-i','--image', required=True,
#                help="path to input image")
#args = vars(ap.parse_args())

## MAIN FUNC BELOW


dir = 'Images/Lateral'

# processAllInFolder(dir)
processImg('Images/Lateral/1556GMZ.jpg')
# processImg('Images/Lateral/3660CRT.jpg')
# oneImg ='3587DCXlp.jpg' 
# img = cv2.imread(oneImg)
# processLpV2(img)
print('processed: ',processed_count,"/  correct: ",correct_count)
if wrong_lp_arr:
    print("License plates incorrectly recognized: " ,wrong_lp_arr)
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







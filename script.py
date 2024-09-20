import argparse
import cv2
import os

dir = 'Images/Frontal'


#ap = argparse.ArgumentParser()
#ap.add_argument('-i','--image', required=True,
#                help="path to input image")
#args = vars(ap.parse_args())

for file in os.listdir(dir):
    f = os.path.join(dir, file)
    print(f)
    if os.path.isfile(f):
        image = cv2.imread(f)
        imS = cv2.resize(image, (1920,1080))


#convert to grayscale
        gImg = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)


#do opening to get rid of noise?
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#opening = cv2.morphologyEx(gImg,cv2.MORPH_OPEN, kernel)

        blurred = cv2.GaussianBlur(gImg, (5,5),0)

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
#rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
#rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,4))
#tophat = cv2.morphologyEx(opening, cv2.MORPH_TOPHAT, rectKernel)
        #blackhat = cv2.morphologyEx(gImg, cv2.MORPH_BLACKHAT, rectKernel)
        #bhb = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, rectKernel)
        edges = cv2.Canny(blurred,150,200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        min_width = 350
        min_height = 100
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:  # Check if the contour has 4 vertices (quadrilateral)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 2.5 < aspect_ratio < 5.0 and w >= min_width and h >= min_height:  # Adjust this range based on the license plate shape
                    license_plate_contour = approx

                    x, y, w, h = cv2.boundingRect(license_plate_contour)
                    print(w,h)
                    license_plate = imS[y:y+h, x:x+w]
                    cv2.rectangle(blurred, (x,y), (x+w,y+h), (0,255,0),3)
                    cv2.imshow("Image",blurred)
                    cv2.waitKey(0)
                    cv2.imshow("Image",edges)
                    cv2.waitKey(0)
                    break





#cv2.imshow("Image",gImg)
#cv2.waitKey(0)
#cv2.imshow("Image",blurred)
#cv2.waitKey(0)
        #cv2.imshow("Image",edges)
        #cv2.waitKey(0)
        #cv2.imshow("Image",imS)
        #cv2.waitKey(0)


cv2.destroyAllWindows()

import argparse
import cv2
import os



def processAllInFolder(dir):
    for file in os.listdir(dir):
        f = os.path.join(dir, file)
        print(f)
        if os.path.isfile(f):
            processImg(f)

def processImg(imgfile):
    img = cv2.imread(imgfile)
    
    imS = cv2.resize(img, (1920,1080))


#convert to grayscale
    gImg = cv2.cvtColor(imS, cv2.COLOR_BGR2GRAY)


#do opening to get rid of noise?
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))

    #blurred = cv2.GaussianBlur(gImg, (5,5),0)
    blurred = cv2.GaussianBlur(gImg, (3,3),0)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
    

    gImg = cv2.erode(gImg, kernel, iterations=1)
    edges = cv2.Canny(blurred,100,150)
    #edges = cv2.Canny(gImg,100,150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("Image",edges)
    #cv2.waitKey(0)
    min_width = 350
    min_height = 80
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Check if the contour has 4 vertices (quadrilateral)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2.5 < aspect_ratio < 6.0 and w >= min_width and h >= min_height:  # Adjust this range based on the license plate shape
                license_plate_contour = approx

                x, y, w, h = cv2.boundingRect(license_plate_contour)
                print(w,h)
                license_plate = imS[y:y+h, x:x+w]
                ## TODO: Once we hace the rectangle of an image, we need to do a Character recognision, to check if there are 4 nums and 3 chars
                ## Also, we can check if there is a small tall rectangle
                cv2.rectangle(imS, (x,y), (x+w,y+h), (0,255,0),3)
                cv2.imshow("Image",imS)
                cv2.waitKey(0)
                cv2.imshow("Image",edges)
                cv2.waitKey(0)
                #break

#ap = argparse.ArgumentParser()
#ap.add_argument('-i','--image', required=True,
#                help="path to input image")
#args = vars(ap.parse_args())

## MAIN FUNC BELOW


dir = 'Images/Frontal'

#processAllInFolder(dir)
processImg('Images/Frontal/5275HGY.jpg')



#cv2.imshow("Image",gImg)
#cv2.waitKey(0)
#cv2.imshow("Image",blurred)
#cv2.waitKey(0)
        #cv2.imshow("Image",edges)
        #cv2.waitKey(0)
        #cv2.imshow("Image",imS)
        #cv2.waitKey(0)


cv2.destroyAllWindows()







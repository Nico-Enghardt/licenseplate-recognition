import cv2
import matplotlib.pyplot as plt
    
def detect_blue(image, ):
    
    saturation_threshold = 150
    
    pixels = image.shape[0]*image.shape[2]
    
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    _, high_saturation_area = cv2.threshold(image_hsv[:,:,1], saturation_threshold,255, cv2.THRESH_BINARY)
    
    high_saturation_hue = high_saturation_area*image_hsv[:,:,0]
    
    hist_values, bins, patches = plt.hist(high_saturation_hue.flatten(), bins=10, range=(3, 255))
    
    hist_values[5] / pixels >= .01
    
    return 
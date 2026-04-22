# Morphological

import numpy as np
import cv2
from PIL import Image

def getaccuracy(imagepath):
    
    image = np.load(imagepath)
    Image.fromarray(image).convert("RGB").save("temp.jpg", quality=95)
    image = cv2.imread("temp.jpg", 0)
    # Erode And Dilate
    kernel1 = np.ones((17, 17), np.uint8)
    kernel2 = np.ones((16, 16), np.uint8)
    image=cv2.erode(image,kernel1)
    image=cv2.dilate(image,kernel2)
    image = cv2.dilate(image, kernel2)
    image = cv2.erode(image, kernel1)
    Image.fromarray(image).save("result/result.png", quality=95)
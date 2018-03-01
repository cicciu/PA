import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils.imgProcessing import *

            
imageFile = "Images/G00368188.jpg"
#imageFile = "Boiss/G00753100.tif"



#Read image
img = cv2.imread(imageFile)
img = cv2.medianBlur(img,5)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transforme img in rgb

ret, thresh = cv2.threshold(imgray, 195, 255, cv2.THRESH_BINARY) #treeshold

titles = ['Original Image', 'Global Thresholding (v = 127)',]
images = [img, thresh]

for i in xrange(2):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

#im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#plt.imshow(img)

"""
#split in rgb
blue,green,red = cv2.split(img)
plt.imshow(red,cmap=plt.cm.gray)

#Hough function to red (threeshold is threeshold of maxima in accumulator)
lines = cv2.HoughLines(red,rho=1,theta=np.pi/180,threshold=130)


img2 = img.copy()
for rho,theta in lines[0]:
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),2)
plt.figure(figsize=(8,6))
plt.imshow(img2)

#image show in gray with red split
#plt.imshow(red,cmap=plt.cm.gray)
"""
plt.show()
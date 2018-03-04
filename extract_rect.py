import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils.imgProcessing import *

            
imageFile = "Images/G00368188.jpg"
#imageFile ="Images/G00009201.jpg"
#imageFile = "Boiss/G00753100.tif"



#Read image
img = cv2.imread(imageFile)

#filters
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transgorm rgb to gray levelb
imgBlur = cv2.medianBlur(imgray,5) #Remove salt and peper
th, img_thresh = cv2.threshold(imgBlur, 195, 255, cv2.THRESH_BINARY) #Treeshold of image


img_floodfill = img_thresh.copy()
 
# Mask used to flood filling.
h, w = img_thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
 
# Floodfill from point (0, 0)  all black in white start in 0,0
cv2.floodFill(img_floodfill, mask, (0,0), 255);
 
# Invert floodfilled image (rect in white)
im_floodfill_inv = cv2.bitwise_not(img_floodfill)
 
# Combine the two images to get the foreground
im_out = img_thresh | im_floodfill_inv



#show images
titles = ['img_thresh', 'img_floodfill','im_floodfill_inv','im_out']
#images = [img, img_thresh]

images = [img_thresh, img_floodfill, im_floodfill_inv, im_out]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

imageFilter = im_out#img_thresh
plt.show()
plt.imshow(imageFilter)


plt.show()
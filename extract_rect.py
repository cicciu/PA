import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils.imgProcessing import *

            
imageFile = "Images/G00368188.jpg"
imageFile ="Images/G00009201.jpg"
#imageFile = "Boiss/G00753100.tif"



#Read image
img = cv2.imread(imageFile)
li = img.shape[0]
col = img.shape[1]


imgF = imgFilterCoutour(img)

plt.imshow(imgF)
plt.show()


rectIm, contoursR = rectImgDetect(imgF, li, col)

plt.imshow(rectIm)
plt.show()

exportRects(img, contoursR, li, col, 10000, 100, 100)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils.imgProcessing import *

            



#imageFile ="Images/G00009201.jpg"
#imageFile = "Images/G00305679.jpg"
#imageFile = "Images/G00368188.jpg"
#imageFile = "Images/G00400015.jpg"
#imageFile = "Images/GDC004054_1.jpg"

#imageFile = "Boiss/G00753100.tif"

#imageFile= "testset/G00753210.jpg"
#imageFile= "testset/G00753213.jpg"
#imageFile= "testset/G00753214.jpg"
imageFile= "testset/G00753215.jpg"
#imageFile= "testset/G00753216.jpg"
imageFile= "testset/G00753911.jpg"
imageFile= "testset/G00753213.jpg"
#imageFile = "testset/G00753220.jpg"

#Read image
img = cv2.imread(imageFile)
li = img.shape[0]
col = img.shape[1]


imgF = emptyRectFilter(img, flagPrint=True)

plt.imshow(imgF)
plt.show()

"""
rectIm, contoursR = rectImgDetect(imgF, li, col)

plt.imshow(rectIm)
plt.show()

exportRects(img, contoursR, li, col, 10000, 100, 100)
"""
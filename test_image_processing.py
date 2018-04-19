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

imageFile= "testset/G00753210.jpg"
#imageFile= "testset/G00753213.jpg"
#imageFile= "testset/G00753214.jpg"
imageFile= "testset/G00753215.jpg"
#imageFile= "testset/G00753216.jpg"
#imageFile= "testset/G00753911.jpg"
imageFile= "testset/G00753213.jpg" #typus
#imageFile = "testset/G00753220.jpg"
imageFile = "testset/G00753876.jpg"
imageFile = "testset/G00753826.jpg"
#imageFile = "datatrain/G00753202.jpg"

#Read image
img = cv2.imread(imageFile)

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800,800)
li = img.shape[0]
col = img.shape[1]


#imgF = whiteRectFilter(img, flagPrint=True)
imgF = whiteRectFilter(img, True)

cv2.imshow('image',imgF)

cv2.waitKey(0)
cv2.destroyAllWindows()


"""
rectIm, contoursR = rectImgDetect(imgF, li, col)

plt.imshow(rectIm)
plt.show()

exportRects(img, contoursR, li, col, 10000, 100, 100)
"""
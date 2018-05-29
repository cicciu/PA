import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils.img_processing import *

#imageFile= "data/databoiss/G00753210.jpg"
#imageFile= "data/databoiss/G00753213.jpg"
#imageFile= "data/databoiss/G00753214.jpg"
#imageFile= "data/databoiss/G00753215.jpg"
#imageFile= "data/databoiss/G00753216.jpg"
imageFile= "data/databoiss/G00753911.jpg"
#imageFile= "data/databoiss/G00753213.jpg" #typus
#imageFile = "data/databoiss/G00753220.jpg"
#imageFile = "data/databoiss/G00753876.jpg"
#imageFile = "data/databoiss/G00753826.jpg"
#imageFile = "data/databoiss/G00753202.jpg"
#imageFile = "data/databoiss/G00798928.jpg"

#Read image
img = cv2.imread(imageFile)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

#show image
cv2.imshow('image',img)
cv2.waitKey(0)

#filter
imgF = rapport(img, True)

#show image filter
cv2.imshow('image',imgF)
cv2.waitKey(0)
cv2.destroyAllWindows()
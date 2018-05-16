import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from utils.imgProcessing import *
import os
import sys
import glob

import dlib
from skimage import io
import matplotlib.pyplot as plt


folder_dataset = "data/dataset_typusrect"
dirNewDataset = "data/dataset_typusrect"


if not os.path.exists(dirNewDataset):
    os.mkdir(dirNewDataset)
    
#image proccesing 
win = dlib.image_window()
for f in glob.glob(os.path.join(folder_dataset, "*.jpg")):
    print("Processing file: {}".format(f))
    #Read image
    img = cv2.imread(f)

    #image processing
    imgF = typusrect_filter(img)

    #export image
    filename = os.path.basename(f)
    cv2.imwrite(os.path.join(dirNewDataset, filename), imgF) 

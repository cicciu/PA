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


folder = "./"

# Now let's run the detector over the images in the imagestmp folder and display the
# results.
print("Showing images...")
win = dlib.image_window()
for f in glob.glob(os.path.join(folder, "*.jpg")):
    print("Processing file: {}".format(f))
    #Read image
    img = cv2.imread(f)
    li = img.shape[0]
    col = img.shape[1]

    #image processing
    imgF = imgFilterCoutour(img)

    #visualize
    """win.clear_overlay()
    win.set_image(imgF)
    dlib.hit_enter_to_continue()"""

    #export image
    cv2.imwrite(os.path.basename(f), imgF) 

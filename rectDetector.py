import os
import sys
import glob

import dlib
from skimage import io
import matplotlib.pyplot as plt

import cv2
import numpy as np
import math
from utils.imgProcessing import *


folder = "testset"


# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector = dlib.simple_object_detector("detect_white_rect.svm")

# We can look at the HOG filter we learned.  It should look like a rect.  Neat!
win_det = dlib.image_window()
win_det.set_image(detector)

# Now let's run the detector over the images in the imagestmp folder and display the
# results.
print("Showing detections on the images in the imagestmp folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    li = img.shape[0]
    col = img.shape[1]


    imgF = imgFilterCoutour(img)

    dets = detector(imgF)
    print("Number of rect detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        
        """out = img[d.top()+1:d.top()+d.height()-1,d.left()+1:d.left()+d.width()-1]
        plt.imshow(out)
        plt.show()"""

    win.clear_overlay()
    win.set_image(imgF)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()


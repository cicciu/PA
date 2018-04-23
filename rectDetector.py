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


folderTestset = "testset"

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector_empty_rect = dlib.simple_object_detector("models/detect_empty_rect.svm")
detector_bar_code = dlib.simple_object_detector("models/detect_bar_code.svm") 
detector_typus_rect = dlib.simple_object_detector("models/detect_typus_rect.svm")
detector_white_rect = dlib.simple_object_detector("models/detect_white_rect.svm") 
# We can look at the HOG filter we learned.  It should look like a rect.  Neat!
"""win_det = dlib.image_window()
win_det.set_image(detector_typus_rect)"""

# Now let's run the detector over the images in the imagestmp folder and display the
# results.
print("Showing detections on the images in the imagestmp folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(folderTestset, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    imgCV2 = cv2.imread(f)
    li = img.shape[0]
    col = img.shape[1]

    #DETECT EMPTY RECT
    imgEmptyRectFilter= emptyRectFilter(imgCV2)
    dets_empty_rect = detector_empty_rect(imgEmptyRectFilter)

    #DETECT BAR CODE
    dets_bar_code = detector_bar_code(imgCV2)

    #DETECT TYPUS RECT
    imgTypusRectFilter = typusRectFilter(imgCV2)
    dets_typus_rect = detector_typus_rect(imgTypusRectFilter)

    #DETECT WHITE RECT
    imgWhiteRectFilter = whiteRectFilter(imgCV2)
    dets_white_rect = detector_white_rect(imgWhiteRectFilter)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets_empty_rect, dlib.rgb_pixel(0,0,0))
    win.add_overlay(dets_bar_code, dlib.rgb_pixel(255,255,255))
    win.add_overlay(dets_typus_rect, dlib.rgb_pixel(255,0,0))
    win.add_overlay(dets_white_rect, dlib.rgb_pixel(255,255,0))
    dlib.hit_enter_to_continue()

    


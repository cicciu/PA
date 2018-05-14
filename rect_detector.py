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


folderTestset = "data/testset"
folderDataboiss = "data/databoiss"

# Now let's use the detector as you would in a normal application.  First we
# will load it from disk.
detector_empty_rect = dlib.simple_object_detector("models/detect_empty_rect.svm")
detector_bar_code = dlib.simple_object_detector("models/detect_bar_code.svm") 
detector_typus_rect = dlib.simple_object_detector("models/detect_typus_rect.svm")
detector_white_rect = dlib.simple_object_detector("models/detect_white_rect.svm")
detector_test = dlib.simple_object_detector("models/rect_vertical.svm")  
# We can look at the HOG filter we learned.  It should look like a rect.  Neat!
"""win_det = dlib.image_window()
win_det.set_image(detector_typus_rect)"""

# Now let's run the detector over the images in the imagestmp folder and display the
# results.
print("Showing detections on the images in the imagestmp folder...")

for f in glob.glob(os.path.join(folderTestset, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    imgCV2 = cv2.imread(f)
    img_rescale = cv2.imread(f)
    li = img.shape[0]
    col = img.shape[1]

    img_rescale = cv2.resize(img_rescale, (col/3, li/3))


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

    #DETECT TEST
    dets_test = detector_test(imgCV2)
   
    #draw rect in image
    im_with_rect = drawRects(imgCV2, dets_test, (0,255,0),3)
    im_with_rect = drawRects(imgCV2, dets_empty_rect, (0,0,0),3)
        
    #path of the big img (important because quality is better for the ocr) 
    img_path = folderDataboiss+'/'+os.path.basename(f)

    #get all image rectangle detect in imagefile
    img_rects = exportRects(dets_test, img_path)

    #texts = readTextsInRects(img_rects)

    #print(texts)


    cv2.imshow('Detection(s) de rectangle(s)', im_with_rect)
    cv2.waitKey(0)


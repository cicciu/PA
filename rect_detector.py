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
import pyzbar.pyzbar as barcode
import pylibdmtx.pylibdmtx as datamatrix


folderTestset = "data/testset"
folderDataboiss = "data/databoiss"

#First we will load detector from disk.
detector_empty_rect = dlib.simple_object_detector("models/emptyrect.svm")
detector_barcode = dlib.simple_object_detector("models/detect_bar_code.svm") 
detector_typus_rect = dlib.simple_object_detector("models/typusrect.svm")
detector_verticalrect = dlib.simple_object_detector("models/rect_vertical.svm")  

#Detector over the images in the imagestmp folder and OCR.
print("Showing detections on the images in the testset folder...")

for f in glob.glob(os.path.join(folderTestset, "*.jpg")):
    #image read
    print("Processing file: {}".format(f))
    img = cv2.imread(f)

    """DETECTION AND DRAW"""

    #emptyrect
    im_emptyrect_filter= emptyrect_filter(img)
    dets_empty_rect = detector_empty_rect(im_emptyrect_filter)
    im_with_rect = drawRects(img, dets_empty_rect, (0,0,0),3)

    #barcode
    dets_barcode = detector_barcode(img)
    im_with_rect = drawRects(img, dets_barcode, (255,255,255), 3)

    #typusrect
    im_typusrect_filter = typusrect_filter(img)
    dets_typusrect = detector_typus_rect(im_typusrect_filter)
    im_with_rect = drawRects(img, dets_typusrect, (0,0,255),3)

    #verticalrect
    dets_verticalrect = detector_verticalrect(img)
    im_with_rect = drawRects(img, dets_verticalrect, (0,255,0),3)  

    #display image with the detection of rect
    cv2.imshow('Detection(s) de rectangle(s)', im_with_rect)
    cv2.waitKey(0)


    #path of the big img (important because quality is better for the ocr and barcode/qrcode reader)
    img_path = folderDataboiss+'/'+os.path.basename(f)

    """READ BARCODE and QRCODE"""
    #get barcode image rectangle detect in imagefile (if we are detect barcode)
    if(len(dets_barcode)==1):
        img_barcode = export_rects(dets_barcode, img_path, False)[0] 
        img_barcode = Image.fromarray(img_barcode) #transform openCV img to PIL image

        print img_path
        codevalue = barcode.decode(img_barcode)
        if codevalue == []:
            print "OLALALALAL"
            codevalue =  datamatrix.decode(img_barcode)
        print codevalue
    
    """OCR"""  
    #get all image rectangle detect in imagefile
    #img_rects = export_rects(dets_verticalrect, img_path)

    #texts = readtexts_in_rects(img_rects)

    #print text detect
    #print(texts)

    


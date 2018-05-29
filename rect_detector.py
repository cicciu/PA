import os
import sys
import glob
import dlib
from skimage import io
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from utils.img_processing import *
import pyzbar.pyzbar as barcode
import pylibdmtx.pylibdmtx as datamatrix
import timeit
import json

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


folder_testset = "data/testset"
folder_databoiss = "data/databoiss"
display_flag = True
images_process_times=[]
json_data={
    "images":[]
}

#First we will load detector from disk.
detector_empty_rect = dlib.simple_object_detector("models/emptyrect.svm")
detector_barcode = dlib.simple_object_detector("models/barecoderect.svm") 
detector_typus_rect = dlib.simple_object_detector("models/typusrect.svm")
detector_horizontalrect = dlib.simple_object_detector("models/rect_horizontal.svm")  

start_time_program = timeit.default_timer()

#Detector over the images in the imagestmp folder and OCR.
print("Detections on the images in the "+folder_testset)
for f in glob.glob(os.path.join(folder_testset, "*.jpg")):
    start_image_process_time = timeit.default_timer()

    img = cv2.imread(f)

    """DETECTION AND DRAW"""

    #emptyrect
    im_emptyrect_filter= emptyrect_filter(img)
    dets_empty_rect = detector_empty_rect(im_emptyrect_filter)
    im_with_rect = draw_rects(img, dets_empty_rect, (0,0,0),3)

    #barcode
    dets_barcode = detector_barcode(img)
    im_with_rect = draw_rects(img, dets_barcode, (255,255,255), 3)

    #typusrect
    im_typusrect_filter = typusrect_filter(img)
    dets_typusrect = detector_typus_rect(im_typusrect_filter)
    im_with_rect = draw_rects(img, dets_typusrect, (0,0,255),3)

    #horizontalrect
    dets_horizontalrect = detector_horizontalrect(img)
    im_with_rect = draw_rects(img, dets_horizontalrect, (0,255,0),3)  

    #path of the big img (important because quality is better for the ocr and barcode/qrcode reader)
    img_path = folder_databoiss+'/'+os.path.basename(f)

    """READ BARCODE and QRCODE"""
    #get barcode image rectangle detect in imagefile (if we are detect barcode)
    if(len(dets_barcode)==1):
        rect_barcode = export_rects(dets_barcode, img_path, False)        
        rect_barcode = Image.fromarray(rect_barcode[0][0]) #transform openCV img to PIL image

        codevalue = barcode.decode(rect_barcode)
        #if the barcode return null result, maybe it's a datamatrix
        if codevalue == []:
            codevalue =  datamatrix.decode(rect_barcode)
        
        
    
    """OCR"""  
    #get all image rectangle detect in imagefile
    rects = export_rects(dets_horizontalrect, img_path, False)
    texts = readtexts_in_rects(rects)

    stop_image_process_time = timeit.default_timer()

    images_process_times.append(stop_image_process_time-start_image_process_time)
    
    #if you want to display info
    if display_flag:
        #image processing
        print("Processing file: {}".format(f))

        #display image with the detection of rect
        cv2.imshow('Detection(s) de rectangle(s)', im_with_rect)
        cv2.waitKey(0)

        #print barcode
        if codevalue !=[]:
            print bcolors.OKBLUE+"Barecode: "+str(codevalue) + bcolors.ENDC
        else:
            print bcolors.FAIL+"Barecode: Are not detected" + bcolors.ENDC

        #print text detect
        if texts != []:
            print bcolors.OKGREEN +"OCR value: "+str(texts) + bcolors.ENDC
        else:
            print bcolors.FAIL+"OCR value: Are not detected" + bcolors.ENDC
    
    #create json object
    new_json_data = create_json_data(os.path.basename(f), rects, texts, codevalue, dets_typusrect, dets_empty_rect)
    json_data["images"].append(new_json_data)


stop_time_program = timeit.default_timer()
program_execution_time = stop_time_program - start_time_program
print "Program execution time: "+str(program_execution_time)+"\r\n" 
print "Mean of image processing: " + str(np.mean(images_process_times))+"\r\n"
print "Standard deviation of image processing: "+ str(np.std(images_process_times))+"\r\n"

#write json file
with open('data_json.json', 'w') as outfile:
    json.dump(json_data, outfile)


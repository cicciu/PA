#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract

SIZE_FACTOR = 3

def emptyrect_filter(img, flagPrint=False):
    #transgorm rgb to gray levelb
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


    #Remove salt and peper
    im_blur = cv2.medianBlur(im_gray,15)

    #Treeshold of image
    th, im_th = cv2.threshold(im_blur, 80, 255, cv2.THRESH_BINARY) 

    if flagPrint:
        #show images
        cv2.imshow('Gray', im_gray)
        cv2.imshow('Blur', im_blur)
        cv2.imshow('Tresh', im_th)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return im_th

def typusrect_filter(img, flagPrint=False):
    # define the list of boundaries

    lower_red = np.array([20, 40, 160])  #GBR
    upper_red = np.array([100,120, 255])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lower_red, dtype = "uint8")
    upper = np.array(upper_red, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    im_filter = cv2.bitwise_and(img, img, mask = mask)

    # Erosion
    kernel = np.ones((3,3), np.uint8)

    # Dilate
    im_dilate = cv2.dilate(im_filter, kernel, iterations=3)

    if flagPrint:
        cv2.imshow('Image', img)
        cv2.imshow('Filter', im_filter)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        
    return im_dilate

def export_rects(dets, img_path,flagPrint=False):
    arr_imgrect = []
    img = cv2.imread(img_path)

    if flagPrint:    
        print("Number of rect detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        #resize top height left width in function of image size
        top = d.top()*SIZE_FACTOR
        height = d.height()*SIZE_FACTOR
        left = d.left()*SIZE_FACTOR
        width = d.width()*SIZE_FACTOR

        #if detection are out of boxe
        if left<0:
            left = 0
        if top<0:
            top = 0

        #read rect in img
        rect_img = img[top:top+height,left:left+width]

        arr_imgrect.append(rect_img)

        if flagPrint:
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            cv2.imshow('Detection(s) de rectangle(s)', rect_img)
            cv2.waitKey(0)

    return arr_imgrect

def readtexts_in_rects(img_rects):
    texts=[]
    for img_rect in img_rects:
        img_rect = Image.fromarray(img_rect)
        #ocr
        text = pytesseract.image_to_string(img_rect)
        texts.append(text)
    return texts

def drawRects(img, dets, color, thickness):
    for k, d in enumerate(dets):
        img = cv2.rectangle(img,(d.left(),d.top()), (d.right(),d.bottom()),color,thickness)
    return img



def test(img,flagPrint=False):
    #Remove salt and peper
    img = cv2.medianBlur(img,7)

    # define the list of boundaries
    lowerBlack = np.array([0, 0, 0])  #GBR
    upperBlack = np.array([159,125, 182])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lowerBlack, dtype = "uint8")
    upper = np.array(upperBlack, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    img_filter_black = cv2.bitwise_and(img, img, mask = mask)


    return img_filter_black

def rapport(img, flagPrint=False):
    #transgorm rgb to gray levelb
    im_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    h, w =  im_raw.shape
    im_raw = cv2.resize(im_raw, (w/3, h/3)) 
    img = cv2.resize(img, (w/3, h/3)) 
    

    # Threshold
    _, im_th1 = cv2.threshold(im_raw, 200, 250, cv2.THRESH_BINARY)
    

    # Erosion
    kernel = np.ones((3,3), np.uint8)
    im_erode = cv2.erode(im_th1, kernel, iterations=1)
    

    # Dilate
    im_dilate = cv2.dilate(im_erode, kernel, iterations=3)
    

    # Canny
    im_canny = cv2.Canny(im_dilate, 240, 250)
    

    """if flagPrint:
        cv2.imshow('Raw', im_raw)
        cv2.imshow('Threshold 1', im_th1)
        cv2.imshow('Erosion', im_erode)
        cv2.imshow('dilate', im_dilate)
        cv2.imshow('Canny', im_canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""

        # find contours (i.e. the 'outlines') in the image and initialize the
    # total number of books found
    _ , contours, hierarchy = cv2.findContours(im_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img2 = img.copy()

    newContoursRect=[]
    totalRectDetect = 0
    # loop over the contours https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    for c in contours:
        # approximate the contour 
        epsilon = 0.1*cv2.arcLength(c,True) #10% epsilon
        approx = cv2.approxPolyDP(c,epsilon, True) 
        
        # if the approximated contour has four points, it is either a square or a rectangle (if 3:triangle)
        if len(approx) == 4:
            cv2.drawContours(img2, [approx], -1, (0, 255, 0), 3) #4 = thickness
            newContoursRect.append(c)
            totalRectDetect += 1

    cv2.imshow("Resultat" , img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img




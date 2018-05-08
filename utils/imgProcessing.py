#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

def emptyRectFilter(img, flagPrint=False):
    #transgorm rgb to gray levelb
    imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


    #Remove salt and peper
    imgBlur = cv2.medianBlur(imGray,15)

    #Treeshold of image
    th, imgThresh = cv2.threshold(imgBlur, 80, 255, cv2.THRESH_BINARY) 

    if flagPrint:
        #show images
        titles = ['imGray', 'imgBlur','imgThresh']
        #images = [img, imgThresh]

        images = [imGray, imgBlur, imgThresh]

        for i in xrange(3):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()

    return imgThresh

def whiteRectFilter(img, flagPrint=False):

    # define the list of boundaries
    lowerWhite = np.array([167, 160, 180])  #GBR
    upperWhite = np.array([255,255, 255])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lowerWhite, dtype = "uint8")
    upper = np.array(upperWhite, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    imgFilterWhite = cv2.bitwise_and(img, img, mask = mask)

    if flagPrint:
        cv2.imshow("image", np.hstack([img, imgFilterWhite]))
        cv2.waitKey(0)

    return imgFilterWhite

def typusRectFilter(img, flagPrint=False):
    # define the list of boundaries

    lowerRed = np.array([20, 40, 160])  #GBR
    upperRed = np.array([80,100, 235])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lowerRed, dtype = "uint8")
    upper = np.array(upperRed, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    imgFilterRed = cv2.bitwise_and(img, img, mask = mask)

    if flagPrint:
        cv2.imshow("image", np.hstack([img, imgFilterRed]))
        cv2.waitKey(0)
    
        
    return imgFilterRed

def exportRects(img, dets, stringDets, flagPrint=False):
    out = []
    if flagPrint:    
        print("Number of tresh " +stringDets + " rect detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        rect_img = img[d.top()+1:d.top()+d.height()-1,d.left()+1:d.left()+d.width()-1]
        out.append(rect_img)

        if flagPrint:
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
            cv2.imshow('image', rect_img)
            cv2.waitKey(0)
            
    return out

def drawRects(img, dets, color, thickness):
    for k, d in enumerate(dets):
        img = cv2.rectangle(img,(d.left()/3,d.top()/3), (d.right()/3,d.bottom()/3),color,thickness)
    return img



def test(img,flagPrint=False):
    li = img.shape[0]
    col = img.shape[1]
    #Remove salt and peper
    img = cv2.medianBlur(img,7)

    # define the list of boundaries
    lowerWhite = np.array([165, 160, 170])  #GBR
    upperWhite = np.array([255,255, 255])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lowerWhite, dtype = "uint8")
    upper = np.array(upperWhite, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    imgFilterWhite = cv2.bitwise_and(img, img, mask = mask)


    # detect edged
    edged = cv2.Canny(imgFilterWhite, 1, 30) #first:threshold 1 second:threshold2

    # construct kernel 
    kernel =np.array([[0,0,1,1,0,0],[0,0,1,1,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,0,0],[0,0,1,1,0,0]], np.uint8)

    #kernel = np.ones((3,3),np.uint8)
    # thicken the edged (dilation)
    dilation = cv2.dilate(edged,kernel,iterations = 1)

    #apply a closing kernel to 'close' gaps between 'white'
    closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    if flagPrint:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.imshow("image",imgFilterWhite)
        cv2.waitKey(0)

    return closed
    
def test2(img,flagPrint=False):
    li = img.shape[0]
    col = img.shape[1]
    #Remove salt and peper
    img = cv2.medianBlur(img,15)

    # define the list of boundaries
    lowerWhite = np.array([160, 160, 170])  #GBR
    upperWhite = np.array([255,255, 255])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lowerWhite, dtype = "uint8")
    upper = np.array(upperWhite, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(img, lower, upper)
    imgFilterWhite = cv2.bitwise_and(img, img, mask = mask)


    # detect edged
    edged = cv2.Canny(imgFilterWhite, 15, 30) #first:threshold 1 second:threshold2

    # construct kernel 
    kernel =np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], np.uint8)

    #kernel = np.ones((3,3),np.uint8)
    # thicken the edged (dilation)
    dilation = cv2.dilate(edged,kernel,iterations = 1)

    #apply a closing kernel to 'close' gaps between 'white'
    closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    if flagPrint:
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.imshow("image",imgFilterWhite)
        cv2.waitKey(0)
    return closed

def test3(img, flagPrint=False):

    #transgorm rgb to gray levelb
    im_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #h, w =  im_raw.shape
    #im_raw = cv2.resize(im_raw, (w/3, h/3)) 
    

    # Threshold
    _, im_th1 = cv2.threshold(im_raw, 200, 250, cv2.THRESH_BINARY)
    

    # Erosion
    kernel = np.ones((3,3), np.uint8)
    im_erode = cv2.erode(im_th1, kernel, iterations=1)
    

    # Dilate
    im_dilate = cv2.dilate(im_erode, kernel, iterations=3)
    

    # Canny
    im_canny = cv2.Canny(im_dilate, 240, 250)
    

    if flagPrint:
        cv2.imshow('Raw', im_raw)
        cv2.imshow('Threshold 1', im_th1)
        cv2.imshow('Erosion', im_erode)
        cv2.imshow('dilate', im_dilate)
        cv2.imshow('Canny', im_canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return im_dilate

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




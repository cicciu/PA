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

def exportRects(img, dets, stringDets):
    print("Number of tresh " +stringDets + " rect detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        
        out = img[d.top()+1:d.top()+d.height()-1,d.left()+1:d.left()+d.width()-1]
        cv2.imshow('image',out)
        cv2.waitKey(0)



def test(img,flagPrint=False):
    #Remove salt and peper
    imgBlur = cv2.medianBlur(img,15)

    # define the list of boundaries
    lowerWhite = np.array([167, 160, 180])  #GBR
    upperWhite = np.array([255,255, 255])  #GBR
    

    # create NumPy arrays from the boundaries
    lower = np.array(lowerWhite, dtype = "uint8")
    upper = np.array(upperWhite, dtype = "uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(imgBlur, lower, upper)
    imgFilterWhite = cv2.bitwise_and(imgBlur, imgBlur, mask = mask)

    if flagPrint:
        cv2.imshow("image", np.hstack([imgBlur, imgFilterWhite]))
        cv2.waitKey(0)

    th, imTh = cv2.threshold(imgFilterWhite, 160, 255, cv2.THRESH_BINARY);

    if flagPrint:
        cv2.imshow("image", np.hstack([img, imTh]))
        cv2.waitKey(0)

    return imgFilterWhite




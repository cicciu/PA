#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imgFilter(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transgorm rgb to gray levelb
    imgBlur = cv2.medianBlur(imgray,5) #Remove salt and peper
    th, img_thresh = cv2.threshold(imgBlur, 195, 255, cv2.THRESH_BINARY) #Treeshold of image


    img_floodfill = img_thresh.copy()
    
    # Mask used to flood filling.
    h, w = img_thresh.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)  all black in white start in 0,0
    cv2.floodFill(img_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image (rect in white)
    im_floodfill_inv = cv2.bitwise_not(img_floodfill)
    
    # Combine the two images to get the foreground
    im_out = img_thresh | im_floodfill_inv

    return img_thresh

def rectImg(img, li, col):
        
    # detect edges in the image
    edged = cv2.Canny(img, 10, 250)


    # construct and apply a closing kernel to 'close' gaps between 'white'
    # pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


    # find contours (i.e. the 'outlines') in the image and initialize the
    # total number of books found
    _ , contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0



    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=lambda  x:x[1],reverse=True)




    #crate new image white
    newImg = np.ones((li, col))

    # loop over the contours
    for c in contours:
        # approximate the contour 
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # if the approximated contour has four points, then assume that the
        # contour is a book -- a book is a rectangle and thus has four vertices
        if len(approx) == 4:
            cv2.drawContours(newImg, [approx], -1, (0, 255, 0), 4)
            total += 1
    return newImg, rects

def exportRects(img, rects, li, col, minArea):
    i = -1
    j = 1
    y_old = li
    x_old = col

    for rect in rects:
        x,y,w,h = rect
        area = w*h
        #test if area is not small
        if area > minArea:
            if (y_old - y) > 200:
                    i += 1
                    y_old = y

            if abs(x_old - x) > 300:
                x_old = x
                x,y,w,h = rect

                out = img[y+10:y+h-10,x+10:x+w-10]
                j+=1
                plt.imshow(out)
                plt.show()
                #export rects
                #cv2.imwrite('cropped\\' + fileName[i] + '_' + str(j) + '.jpg', out)

            




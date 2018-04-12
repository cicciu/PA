#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imgFilterCoutour(img):
    #transgorm rgb to gray levelb
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


    #Remove salt and peper
    imgBlur = cv2.medianBlur(imgray,5)

    #Treeshold of image
    th, img_thresh = cv2.threshold(imgBlur, 195, 255, cv2.THRESH_BINARY) 

    # detect edged
    edged = cv2.Canny(img_thresh, 1, 255) #first:threshold 1 second:threshold2

    # construct kernel 
    kernel =np.array([[0,0,1,1,0,0],[0,0,1,1,0,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,0,1,1,0,0],[0,0,1,1,0,0]], np.uint8)

    #kernel = np.ones((3,3),np.uint8)
    # thicken the edged (dilation)
    dilation = cv2.dilate(edged,kernel,iterations = 1)

    #apply a closing kernel to 'close' gaps between 'white'
    closed = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)


    #show images
    titles = ['imgray', 'imgBlur','img_thresh','edged']
    #images = [img, img_thresh]

    images = [imgray, imgBlur, img_thresh, edged]

    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    #show images
    titles = [ 'dilation', 'closed']
    #images = [img, img_thresh]

    images = [dilation, closed]

    for i in xrange(2):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


    return closed

def rectImgDetect(img, li, col):
    # find contours (i.e. the 'outlines') in the image and initialize the
    # total number of books found
    _ , contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #crate new image white
    newImg = np.ones((li, col))

    newContoursRect=[]
    totalRectDetect = 0
    # loop over the contours https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    for c in contours:
        # approximate the contour 
        epsilon = 0.01*cv2.arcLength(c,True) #10% epsilon
        approx = cv2.approxPolyDP(c,epsilon, True) 
        
        # if the approximated contour has four points, it is either a square or a rectangle (if 3:triangle)
        if len(approx) == 4:
            cv2.drawContours(newImg, [approx], -1, (0, 255, 0), 10) #4 = thickness
            newContoursRect.append(c)
            totalRectDetect += 1

    return newImg, newContoursRect


def exportRects(img, contours, li, col, minArea, minW, minH):
    imgNum = 0
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    #sort order of rects (reverse false:Bottom-Up true:Up-Bottom )
    rects = sorted(rects,key=lambda  x:x[1],reverse=False) 

    for rect in rects:
        x,y,w,h = rect
        area = w*h
        #test if area is not small
        if w > minW and h > minH and area > minArea:
            x,y,w,h = rect

            out = img[y+1:y+h-1,x+1:x+w-1]
            
            plt.imshow(out)
            plt.show()
            #export rects
            #cv2.imwrite('cropped\\' + str(imgNum) + '.jpg', out)
            #imgNum+=1

            




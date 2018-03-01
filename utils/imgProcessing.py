#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
from PIL import Image
import sys
from math import sqrt
import numpy as np
    

def invert(image):
    """this function make a negative image

    Prams :
    - -- -
    """
    try:
        img = Image.open(ImageFile)
    except IOError:
        print 'Erreur sur ouverture du fichier ' + ImageFile

    newImage = Image.new(img.mode,img.size)

    col,li = img.size

    for i in range(li):
        for j in range(col):
            pixel = img.getpixel((j,i)) # recup du pixel
            # on calcule le complement a MAX pour chaque composante - effet negatif
            p = (255 - pixel[0], 255 - pixel[1], 255 - pixel[2])
            # composition de la nouvelle image
            newImage.putpixel((j,i), p)
    return newImage

def contourDetection(seuil, image):
    newImage = image.new(img.mode,img.size)

    col,li = img.size

    for i in range(1,li-1):
        for j in range(1,col-1):
            p1 = img.getpixel((j-1,i))
            p2 = img.getpixel((j,i-1))
            p3 = img.getpixel((j+1,i))
            p4 = img.getpixel((j,i+1))
            n = norme(p1,p2,p3,p4)

            if n < seuil:
                p = (255,255,255)
            else:
                p = (0,0,0)
            newImage.putpixel((j-1,i-1),p)
    return newImage
def norme(self,p1,p2,p3,p4):
    n = sqrt((p1[0]-p3[0])*(p1[0]-p3[0]) + (p2[0]-p4[0])*(p2[0]-p4[0]))
    return n
"""
def convolution2D(self,filt,TPix,x,y):
    p0 = p1 = p2 = 0
    for i in range(-1,1):
        for j in range(-1,1):
            p0 += filt[i+1][j+1]*TPix[y+i,x+j][0]
            p1 += filt[i+1][j+1]*TPix[y+i,x+j][1]
            p2 += filt[i+1][j+1]*TPix[y+i,x+j][2]
            # normalisation des composantes
            p0 = int(p0/9.0)
            p1 = int(p1/9.0)
            p2 = int(p2/9.0)
    # retourne le pixel convolué
    return (p0,p1,p2)

def filter(self,filt):
    newImage = Image.new(self.imageProcess.mode,self.imageProcess.size)
    TabPixel = self.imageProcess.load()
    for x in range(1,self.ligne-1):
        for y in range(1,self.colonne-1):
            p = self.convolution2D(filt,TabPixel,x,y)
            newImage.putpixel((y,x),p)
    self.imageProcess = newImage
    return newImage


"""



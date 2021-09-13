# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:00:42 2021

@author: talha
"""


# let say there is an image whose pixels values are in the range , meaning that it is not spread in all values.

import numpy as np
import cv2 

import sys
sys.path.append('../4-Utils/')
import helper

#path 
path  = "../0-ReadMe_Files/image.jfif"

# convert lab value into l , a and b , l stands for lightness .
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(img)

#TRADITIONAL WAY _______________________________________________________________________________
#take contrast value
contra_img = l

#show and save
helper.show_img(contra_img,axis="off",save="L Image.jpg",title="L Image")


#hist value takes how many pixel have same value correspont to the bins.
hist,bins = np.histogram(contra_img.flatten(),256,[0,256])


#y_min and y_max value are importatnt
#total value will be contra_img.height *contra_img.witdh
cdf = hist.cumsum()

#show and save
helper.show_plt(cdf,axis="on",save="Raw Image Histogram.jpg",title="Raw Image Histogram")

# put the value 0-255 range
cdf_m = (cdf - cdf.min())*255/(cdf.max()-cdf.min())

#make it integer because images have integer value.
cdf_k = cdf_m.astype('uint8')

#equalized
img2 = cdf_k[contra_img]

#show and save
helper.show_img(img2,axis="off",save="Equalized L Image.jpg",title="Equalized L Image")

#qualized histogram
hist,bins = np.histogram(img2.flatten(),255,[0,255])
cdf_2 = hist.cumsum()
helper.show_plt(cdf_2,axis="on",save="Equalized Histogram.jpg",title="Equalized Histogram")

#merge all l, a and b
updated_img = cv2.cvtColor(cv2.merge((img2, a, b)), cv2.COLOR_LAB2BGR)

#show and save

helper.show_img(updated_img,axis="off",save="Traditional Equalized  Image.jpg",title="Traditional Equalized  Image")

#___________________________________________________________________________________________________

#Histogram Equalization WAY _______________________________________________________________________________

equ = cv2.equalizeHist(l)
updated_lab_img1 = cv2.merge((equ,a,b))
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

#show and save
helper.show_img(hist_eq_img,axis="off",save="EqualizeHist Image.jpg",title="EqualizeHist Image")

#___________________________________________________________________________________________________



#Adaptive Equalization WAY _______________________________________________________________________________
from skimage import exposure

img_adapteq = exposure.equalize_adapthist(l, clip_limit=0.1)
img_adapteq = (img_adapteq *255).astype('uint8')
updated_lab_img2 = cv2.merge((img_adapteq,a,b))
img_adapteq = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
helper.show_img(img_adapteq,axis="off",save="Adaptive Equalized Image.jpg",title="Adaptive Equalized Image")

#___________________________________________________________________________________________________


#Contrast stretching WAY _______________________________________________________________________________

p2, p92 = np.percentile(l, (2, 92))
img_rescale = exposure.rescale_intensity(l, in_range=(p2, p92))
updated_lab_img2 = cv2.merge((img_rescale,a,b))
contrast_stretch = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
helper.show_img(contrast_stretch,axis="off",save="Contrast Stretched Image.jpg",title="Contrast Stretched Image Image")

#___________________________________________________________________________________________________

# CLAHE  WAY _______________________________________________________________________________

clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(12,12))
clahe_img = clahe.apply(l)
updated_lab_img2 = cv2.merge((clahe_img,a,b))
clahe_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
helper.show_img(clahe_img,axis="off",save="CLAHE Image.jpg",title="CLAHE Image")

#___________________________________________________________________________________________________


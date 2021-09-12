# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 20:00:42 2021

@author: talha
"""


# let say there is an image whose pixels values are in the range , meaning that it is not spread in all values.

import numpy as np
import cv2 
from matplotlib import pyplot as plt

path  = "../0-ReadMe_Files/img4.jfif"

# convert hsv image contains value , hue  and saturation .
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(img)


#take contrast value
contra_img = l
plt.imshow(contra_img)
plt.title("Raw Image")
plt.show()

#hist value takes how many pixel have same value correspont to the bins.
hist,bins = np.histogram(contra_img.flatten(),256,[0,256])


#y_min and y_max value are importatnt
#total value will be contra_img.height *contra_img.witdh
cdf = hist.cumsum()
plt.plot(cdf, color = 'b')
plt.show()

# put the value 0-255 range
cdf_m = (cdf - cdf.min())*255/(cdf.max()-cdf.min())

#make it integer because images have integer value.
cdf_k = cdf_m.astype('uint8')


img2 = cdf_k[contra_img]
plt.imshow(img2)
plt.title("Equalized Image")
plt.show()
hist,bins = np.histogram(img2.flatten(),255,[0,255])
cdf_2 = hist.cumsum()
plt.plot(cdf_2, color = 'b')
plt.show()

updated_img2 = cv2.merge((img2, a, b))
plt.imshow(cv2.cvtColor(updated_img2, cv2.COLOR_LAB2BGR))
plt.axis("off")
plt.show()

############
equ = cv2.equalizeHist(l)
updated_lab_img1 = cv2.merge((equ,a,b))
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)
plt.imshow(hist_eq_img)
plt.axis("off")
plt.show()
##########

#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(12,12))
clahe_img = clahe.apply(l)
updated_lab_img2 = cv2.merge((clahe_img,a,b))
CLAHE_img2 = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
cv2.imwrite("./denem11e.jpg", CLAHE_img2)
plt.imshow(CLAHE_img2)
plt.axis("off")
plt.show()
##########
# Adaptive Equalization
from skimage import exposure

img_adapteq = exposure.equalize_adapthist(l, clip_limit=0.03)
img_adapteq = (img_adapteq *255).astype('uint8')
updated_lab_img2 = cv2.merge((img_adapteq,a,b))
CLAHE_img2 = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
cv2.imwrite("./denemtte.jpg", CLAHE_img2)
plt.imshow(CLAHE_img2)
plt.axis("off")
plt.show()

##########
# Contrast stretching
p2, p98 = np.percentile(l, (2, 100))
img_rescale = exposure.rescale_intensity(l, in_range=(p2, p98))
updated_lab_img2 = cv2.merge((img_rescale,a,b))
CLAHE_img2 = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
plt.imshow(CLAHE_img2)
plt.axis("off")
plt.show()

#yuv lab image
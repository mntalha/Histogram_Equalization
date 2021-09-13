# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:15:33 2021

@author: talha
"""

from matplotlib import pyplot as plt

def show_imgs(original , converted, axis = None,save="test.jpg",img1="Normal Image",img2="Transformed Image"):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.title(img1)
    plt.imshow(original)
    plt.axis(axis)
    fig.add_subplot(1,2,2)
    plt.title(img2)
    plt.imshow(converted)
    plt.axis(axis)
    plt.show()
    fig.savefig("../3-Results"+save,dpi = 250) #dpi --> high resolution

def show_img(original,axis = None,save="test.jpg",title="Normal Image"):
    fig = plt.figure()
    fig.add_subplot(1,1,1)
    plt.title(title)
    plt.imshow(original)
    plt.axis(axis)
    plt.show()
    fig.savefig("../3-Results/"+save,dpi = 250) #dpi --> high resolution

def show_plt(original,axis = None,save="test.jpg",title="Normal Image"):
    fig = plt.figure()
    fig.add_subplot(1,1,1)
    plt.title(title)
    plt.plot(original,color = 'b')
    plt.axis(axis)
    plt.show()
    fig.savefig("../3-Results/"+save,dpi = 250) #dpi --> high resolution

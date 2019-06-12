import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage.util import img_as_float
from skimage.feature import greycomatrix, greycoprops
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd
import cv2
import os

root_dir_test = '../VOCdevkit2/VOC2007/'
img_dir_test = os.path.join(root_dir_test, 'JPEGImages')
ann_dir_test = os.path.join(root_dir_test, 'Annotations')
gt_dir_test = os.path.join(root_dir_test, 'SegmentationClass')
set_dir_test = os.path.join(root_dir_test, 'ImageSets', 'Main')


root_dir = '../VOCdevkit/VOC2007/'
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Segmentation')

labels = ['bg', 'aeroplane', 'bicycle', 'bird', 
         'boat', 'bottle', 'bus', 'car', 
         'cat', 'chair', 'cow', 'diningtable', 
         'dog', 'horse', 'motorbike', 'person', 
         'pottedplant', 'sheep', 'sofa', 
         'train', 'tvmonitor'] 

def printOutput(output,segments, pred):
    for (i, segVal) in enumerate(np.unique(segments)):
        output[segments == segVal] = pred[i]
    return output

def analiseSuperpixels(image):
    #image = img_as_float(image[::2, ::2])
    #GT = cv2.imread()
    segments = slic(img_as_float(image), n_segments = 100, sigma = 5)

    for (i, segVal) in enumerate(np.unique(segments)):
        (B, G, R) = cv2.split(image.astype("float"))
	    # construct a mask for the segment
        mask = np.ones(image.shape[:2], dtype = "uint8")
        mask[segments == segVal] = 0        
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(G, mask=mask)
        B = np.ma.masked_array(B, mask=mask)
        seg = cv2.merge((B,G,R))
        seg = seg.astype("uint8")
        seg = cv2.bitwise_and(image, image, mask = mask)
        props = comatImg(seg) 
        if (i==0):
            testProps = props
        else:
            testProps = np.vstack((testProps, props))
        
    return testProps, segments
def begTest(X_train, Y_train):
    clf_svm = SVC(gamma='scale')
    clf_svm.fit(X_train, Y_train)

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, Y_train)

    return clf_svm, clf_rf

def test(image, name, clf_rf, clf_svm):
    heigh, width, _ = image.shape
    saida1 = np.zeros((heigh, width))
    saida2 = np.zeros((heigh, width))

    testProps, segss = analiseSuperpixels(image)
    pred_svm = clf_svm.predict(testProps)
    
    saida1 = printOutput(saida1, segss,pred_svm)
    np.savetxt(name.replace('.jpg', '') + '_1.txt', saida1, fmt='%d')

    pred_svm = clf_rf.predict(testProps)
    
    saida2 = printOutput(saida2,segss,pred_svm)
    np.savetxt(name.replace('.jpg', '') + '_2.txt', saida2, fmt='%d')

def comatImg(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    d = 1
    matrix = greycomatrix(gray, [d], [0], normed=True)

    props = np.zeros((6))
    props[0] = greycoprops (matrix, 'contrast')
    props[1] = greycoprops (matrix, 'dissimilarity')
    props[2] = greycoprops (matrix, 'homogeneity')
    props[3] = greycoprops (matrix, 'energy')
    props[4] = greycoprops (matrix, 'correlation')
    props[5] = greycoprops (matrix, 'ASM')
    return props

def filtering_image(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    inv_sure_bg = cv2.bitwise_not(sure_bg)
    res = cv2.bitwise_and(img, img, mask = sure_bg)
    bg = cv2.bitwise_and(img, img, mask = inv_sure_bg)
    return res, bg


def segmentations(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img_as_float(img[::2, ::2])
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    print_and_plot(img, segments_fz, "Felzenszwalb", ax)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    print_and_plot(img, segments_slic, "SLIC", ax)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    print_and_plot(img, segments_quick, "Quickshift", ax)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    print_and_plot(img, segments_watershed, "Compact Watershed", ax)
    
    #for a in ax.ravel():
    #    a.set_axis_off()
    #plt.tight_layout()
    #plt.show()

    return segments_fz
    

def print_and_plot(img, seg, name, ax):
    i = 0
    j = 0
    if(name == "Compact Watershed" or name == "Quickshift"):
        i = 1
    if(name == "Compact Watershed" or name == "SLIC"):
        j = 1
    print(name + " number of segments: {}".format(len(np.unique(seg))))
    ax[i, j].imshow(mark_boundaries(img, seg))
    ax[i, j].set_title(name + "'s method")

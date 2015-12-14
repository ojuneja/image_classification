#!/usr/bin/python

# import cv2
# from sklearn.feature_extraction.text import TfidfVectorizer
# #from sklearn.cluster import KMeans
# import setfeatures
# import normalizefraction
# import variables as var
# import csv
# import imutils 

import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# command line argument parse to get input directory
if len(sys.argv) == 2:
    train_path = sys.argv[1]
else:
    print 'Give valid command line arguments'
    sys.exit(1)
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
class_id = 0
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = dir
    image_paths.append(dir)

# Create feature extraction and keypoint detector objects
sift = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    #kpts = fea_det.detect(im)
    #kpts, des = des_ext.compute(im, kpts)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (kps, descs) = sift.detectAndCompute(gray, None)
    des_list.append((image_path, descs))   

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))
    #print descriptors

# Perform k-means clustering
k =  500
value = []
klist = []
count = 10
while count <= k:
    voc, variance = kmeans(descriptors, count, 1)
    print count, variance
    value.append(variance)
    klist.append(count)
    count = count+10

plt.plot(klist,value) 
print "plotting graph"
plt.show()
#!/usr/bin/python
import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import imutils 

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


# command line argument parse to get input directory
if len(sys.argv) == 2:
    train_path = sys.argv[1]
else:
    print 'Give valid command line arguments'
    sys.exit(1)
print train_path
training_names = os.listdir(train_path)


# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []

for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    #print dir
    #print type(dir) 
    class_path = dir#imutils.imlist(dir)
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
k = 50
print k
voc, variance = kmeans(descriptors, k, 1)

# Calculate the histogram of features
im_features = np.zeros( (len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)


print len(im_features)/2
image_classes = [0] * 1000
image_classes+= [1] * 1000

print image_classes
# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))
classes = ['d','a']


# Save the SVM
joblib.dump((clf, classes, stdSlr, k, voc), "model_large.pkl", compress=3)
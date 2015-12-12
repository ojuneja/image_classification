#!/usr/bin/python
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import setfeatures
import normalizefraction
import csv



k = 10
vocab = []              # This is a list of every SIFT descriptor.
raw_corpus = []
imp = []
sift = cv2.xfeatures2d.SIFT_create()
mypath = './minidataset'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for image in onlyfiles:
        image = mypath+ '/' + image
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp = sift.detect(gray,None)
	#img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imwrite(image+'sift_keypoints.jpg',img)
        i = 1
        kp,desc = sift.compute(gray,kp)
	#print desc
	if desc != None:
                        img_features = []
                        for row in desc:
                                        i+=1
                                        vocab.append(row.tolist())
                                        img_features.append(row.tolist())
                        raw_corpus.append(img_features)
                        imp.append(i)

#print raw_corpus
# Perform clustering with k clusters. This will probably need tuning.
cluster = KMeans(k, n_init=1)
cluster.fit(vocab)
#print minicorpus
# Now we build the clustered corpus where each entry is a string containing the cluster ids for each sift-feature.
corpus = []
for entry in raw_corpus:
                        corpus.append(' '.join([str(x) for x in cluster.predict(entry)]))
#print corpus
#now we are setting our features and thereby normalizing our values
#print setfeatures.setFeatures(corpus,k)
table =  normalizefraction.normalize(setfeatures.setFeatures(corpus,k),k)
#print table
with open("output.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(table)



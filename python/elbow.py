#!/usr/bin/python
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


k = 20
sum = 0
vocab = []              # This is a list of every SIFT descriptor.
raw_corpus = []
imp = []
plotdata = []
sift = cv2.xfeatures2d.SIFT_create()
mypath = '/usr/local/lib/python2.7/site-packages/miniminidataset'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
count = 1
while count <= k:
	for image in onlyfiles:
        	image = mypath+ '/' + image
	        img = cv2.imread(image)
        	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	        kp = sift.detect(gray,None)
        	i = 1
	        kp,desc = sift.compute(gray,kp)
        	if desc != None:
        	                img_features = []
                	        for row in desc:
                        	                i+=1
                                	        vocab.append(row.tolist())
                                        	img_features.append(row.tolist())
				raw_corpus.append(img_features)
				
	#print raw_corpus		
	# Perform clustering with k clusters. This will probably need tuning.
	cluster = KMeans(count, n_init=1)
	cluster.fit(vocab)
	# Now we build the clustered corpus where each entry is a string containing the cluster ids for each sift-feature.
	corpus = []
	for entry in raw_corpus:
				corpus.append(' '.join([str(x) for x in cluster.predict(entry)]))
	distance = cluster.inertia_
	plotdata.append(distance)
	sum = sum + distance
	count+=1
list = []
average = sum / k
for val in plotdata: list.append(abs(average-val))
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
plt.scatter(list, y)
plt.show()
#plt.bar(range(0,max(list)), list)
#plt.show()


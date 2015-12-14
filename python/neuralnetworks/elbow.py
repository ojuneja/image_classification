#!/usr/bin/python
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



vocab = []              # This is a list of every SIFT descriptor.
raw_corpus = []
imp = []
plotdata = []
sift = cv2.xfeatures2d.SIFT_create()
mypath ='./traincar'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print onlyfiles
for image in onlyfiles:
        	image = mypath+ '/' + image
	        img = cv2.imread(image)
        	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	        kp = sift.detect(gray,None)
        	kp,desc = sift.compute(gray,kp)
        	if desc != None:
        	                img_features = []
                	        for row in desc:
                                                vocab.append(row.tolist())
                                        	img_features.append(row.tolist())
				raw_corpus.append(img_features)
				
#print raw_corpus		
# Perform clustering with k clusters. This will probably need tuning.
plotdata = []
k_values = []
count = 60
k = 20
while k <= count:
	cluster = KMeans(k, n_init=1)
	cluster.fit(vocab)
# Now we build the clustered corpus where each entry is a string containing the cluster ids for each sift-feature.
	corpus = []
	for entry in raw_corpus:
				corpus.append(' '.join([str(x) for x in cluster.predict(entry)]))


	print "calculating value of k"	
	#plt.scatter(list, plotdata)
	print "value of k: ",k
	k_values.append(k)
	plotdata.append(cluster.inertia_)
	k+=2
#plt.show()
plt.plot(k_values,plotdata)
plt.show()


#!/usr/bin/python
from __future__ import division
import numpy as np
import variables as var
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from sklearn.metrics import mean_squared_error as MSE

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

# load model
net = pickle.load( open(var.output_model_file, 'rb' ))
#load data
test = np.loadtxt( var.test_file, delimiter = ',' )
input_data = test[:,0:-1]
target_data = test[:,-1]
target_data = target_data.reshape( -1, 1 )
#print input_data,target_data
# prepare dataset
ds = SDS( var.no_of_clusters, var.output )
ds.setField( 'input', input_data )
ds.setField( 'target', target_data )
#activate network
predict_list = net.activateOnDataset(ds)	
for predict,ground_truth in zip(predict_list,target_data):
	if predict <= 0.0:
			if ground_truth <= 0 : true_negative += 1
			else: false_negative += 1
			print "Pedicted: NOT Car"
	else : 
		if ground_truth <= 0 : false_positive += 1
		else: true_positive += 1
		print "Predicted: Car"
#print true_positive,true_negative,false_positive,false_negative
precision =  true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
#print precision,recall
fscore = 2 * ((precision * recall) / (precision + recall))
print "FSCORE: " + str(fscore)
#predict results and calculate RMSE
#mse = MSE( target_data , p )
#rmse = sqrt(mse)
#np.savetxt( "prediction.txt", p, fmt = '%.6f' )
#print "testing RMSE:", rmse

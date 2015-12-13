#!/usr/bin/python
import numpy as np
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from sklearn.metrics import mean_squared_error as MSE


output_model_file = 'model_val.pkl'
test_file = 'test.csv'

# load model
net = pickle.load( open(output_model_file, 'rb' ))
#load data
test = np.loadtxt( test_file, delimiter = ',' )
input_data = test[:,0:-1]
target_data = test[:,-1]
target_data = target_data.reshape( -1, 1 )
#print input_data,target_data
# prepare dataset
ds = SDS( 10, 1 )
ds.setField( 'input', input_data )
ds.setField( 'target', target_data )
#activate network
p = net.activateOnDataset(ds)	
#print p
#predict results
mse = MSE( target_data , p )
rmse = sqrt(mse)
np.savetxt( "prediction.txt", p, fmt = '%.6f' )
print "testing RMSE:", rmse

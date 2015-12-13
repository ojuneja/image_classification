#!/usr/bin/python
import pybrain
import numpy as np
import cPickle as pickle
from pybrain.structure.modules import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


output_model_file = 'model_val.pkl'
epochs = 600
continue_epochs = 10
validation_proportion = 0.15

#fix iputs and outputs of dataset
ds = SupervisedDataSet(10,1)
tf = open('output.csv','r')
#take input and output from .csv file and set input and output
for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[:10])
    outdata = tuple(data[10:])
    ds.addSample(indata,outdata)
#print ds.indim, ds.outdim
#buil network with backpropagation
net = buildNetwork(ds.indim,2,ds.outdim)
trainer = BackpropTrainer(net,ds)
#train newtwork and see error
#print trainer.train()
train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = validation_proportion, 
	maxEpochs = epochs, continueEpochs = continue_epochs )
#print train_mse, validation_mse
pickle.dump( net, open( output_model_file, 'wb' ))


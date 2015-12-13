#!/usr/bin/python
import pybrain
import numpy as np
import cPickle as pickle
import variables as var
from pybrain.structure.modules import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.metrics import mean_squared_error as MSE
from math import sqrt



#fix iputs and outputs of dataset
ds = SupervisedDataSet(var.no_of_clusters,var.output)
tf = open(var.output_file,var.r)
#take input and output from .csv file and set input and output
for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[:var.no_of_clusters])
    outdata = tuple(data[var.no_of_clusters:])
    ds.addSample(indata,outdata)
#print ds.indim, ds.outdim
#buil network with backpropagation
net = buildNetwork(ds.indim,var.hidden_layer,ds.outdim)
trainer = BackpropTrainer(net,ds)
#train newtwork and see error
#print trainer.train()
#mse = trainer.train()
#rmse = sqrt( mse )
#print "training RMSE, epoch {}: {}".format( 1, rmse )
train_mse, validation_mse = trainer.trainUntilConvergence( verbose = True, validationProportion = var.validation_proportion,
	maxEpochs = var.epochs, continueEpochs = var.continue_epochs )
#print train_mse, validation_mse
pickle.dump( net, open( var.output_model_file, var.wb ))


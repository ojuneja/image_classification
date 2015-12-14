#!/usr/bin/python


#set this configuration to generate normalized features for train airplane
'''path = './traincar'
out = 1
option = 'wb'
outputfile =  'output.csv' '''

#set this configuration to generate normalized features for train not airplane
'''path = './trainbird'
out = -1
option = 'a'
outputfile =  'output.csv' '''

#set this configuration to generate normalized features for test not airplane
'''path = './testbird'
out = -1
option = 'wb'
outputfile =  'test.csv' '''

#set this configuration to generate normalized features for test airplane
path = './testcar'
out = 1
option = 'a'
outputfile =  'test.csv'


#variables of test py files
no_of_clusters = 100
output = 1


#variables of train and testdata py files
output_file =  'output.csv' 
test_file =  'test.csv' 
epochs = 1000
continue_epochs = 10
validation_proportion = 0.15
hidden_layer = 2
wb = 'wb'
r = 'r'


#output model file
output_model_file = 'model.pkl'



#!/usr/bin/python


#set this configuration to generate normalized features for train airplane
'''path = './trainairplane'
out = 1
option = 'wb'
outputfile =  'output.csv' '''

#set this configuration to generate normalized features for train not airplane
'''path = './trainnotairplane'
out = 0
option = 'a'
outputfile =  'output.csv' '''

#set this configuration to generate normalized features for test not airplane
'''path = './testnotairplane'
out = 0
option = 'wb'
outputfile =  'test.csv' '''

#set this configuration to generate normalized features for test airplane
'''path = './testairplane'
out = 1
option = 'a'
outputfile =  'test.csv' '''



#variables of test py files
no_of_clusters = 10
output = 1


#variables of train and testdata py files
output_file =  'output.csv' 
test_file =  'test.csv' 
epochs = 600
continue_epochs = 10
validation_proportion = 0.25
hidden_layer = 2
wb = 'wb'
r = 'r'


#output model file
output_model_file = 'model.pkl'



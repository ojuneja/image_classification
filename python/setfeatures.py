#!/usr/bin/python

def setFeatures(corpus,k):
	colldictionary = []
	for entry in corpus:
        	                arraywords = entry.split()
        	                dictionary = {}
        	                counter = 0
        	                while counter < k:
        	                                        dictionary['' + str(counter)] = 0;
        	                                        counter+=1
        	                for word in arraywords:
        	                                        #print dictionary.get(word)
        	                                        dictionary[word] = dictionary.get(word) + 1;
        	                colldictionary.append(dictionary)
	return colldictionary


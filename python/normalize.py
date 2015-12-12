#!/usr/bin/python
import numpy as np
def normalize(colldictionary,k):
		table = []
		normalizeTable = []
		counter = 0
		size = len(colldictionary)
		while counter < k:
                		normalizeList = []
				sum = 0
                		list = []
                		for item in colldictionary:
                		                list.append(item.get('' + str(counter)))
                		                sum = sum + item.get('' + str(counter))
                		stdDev = np.std(list)
                		average = sum / size;
                		for value in list:
                		                normalizeValue = abs(value - average) / stdDev
                		                normalizeList.append(normalizeValue)
                		counter+=1
	        		table.append(normalizeList)
		counter = 0
		while counter < size:
					count = 0
					normalizeList = []
					normalizeList.append('id' + str(counter))
					while count < k:
							normalizeList.append(table[count][counter])
							count+=1
					normalizeList.append('airplane')
					normalizeTable.append(normalizeList)
					counter+=1
		return normalizeTable


	

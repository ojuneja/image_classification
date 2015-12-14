#!/usr/bin/python
import numpy as np
import variables as var

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
						if stdDev != 0:
                		                	normalizeValue = (value - average) / stdDev
							if(normalizeValue > 2) : normalizeValue = 0.0
							if(normalizeValue < -2): normalizeValue = 0.0
	                		                normalizeList.append(normalizeValue)
						else : normalizeList.append(0.0)
                		counter+=1
	        		table.append(normalizeList)
		counter = 0
		while counter < size:
					count = 0
					normalizeList = []
					#normalizeList.append(counter)
					while count < k:
							normalizeList.append(table[count][counter])
							count+=1
					normalizeList.append(var.out)
					normalizeTable.append(normalizeList)
					counter+=1
		return normalizeTable


	

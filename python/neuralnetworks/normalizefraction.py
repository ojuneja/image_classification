#!/usr/bin/python
from __future__ import division
import numpy as np
import variables as var


def normalize(listDictionary,k):
				table = []
				i = 0
				for singlelistDictionary in listDictionary:
								normalizeList = []
								#normalizeList.append(i)
								sumColl = sum(singlelistDictionary.values())
								for value in singlelistDictionary.values():
											normalizeList.append(value/sumColl)
								normalizeList.append(var.out)
								i+=1			
								table.append(normalizeList)
				return table
'''def normalize(listDictionary,k):
				table = []
				i = 0
				for singlelistDictionary in listDictionary:
								normalizeList = []
								#normalizeList.append(i)
								sumColl = sum(singlelistDictionary.values())
								average = sumColl / len(singlelistDictionary)
								stdDev = np.std(singlelistDictionary.values())
								for value in singlelistDictionary.values():
											normalizeList.append((average - value ) / stdDev)
								normalizeList.append(var.out)
								i+=1			
								table.append(normalizeList)
				return table '''

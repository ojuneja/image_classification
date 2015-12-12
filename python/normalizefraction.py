#!/usr/bin/python
from __future__ import division
import numpy as np


def normalize(listDictionary,k):
				table = []
				i = 0
				for singlelistDictionary in listDictionary:
								normalizeList = []
								normalizeList.append('img' + str(i))
								sumColl = sum(singlelistDictionary.values())
								for value in singlelistDictionary.values():
											normalizeList.append(value/sumColl)
								normalizeList.append('airplane')
								i+=1			
								table.append(normalizeList)
				return table

#!/usr/bin/env python3

"""
File:	frequency_threshold_filter.py
Author:	Ivo Taris, s2188724

This script takes the tokenized reviews_propname as input
and creates a pickle file that features a bag of words,
in which the enlisted words occur at least 5 times.
"""

import os
import pickle
from collections import Counter
from collections import defaultdict
import re

# get the current working directory
cwd = os.getcwd()

# the absolute path to the totaltrainset
directory = cwd + '/reviews_propname/alldata1/'	

words_all = []

for file in os.listdir(directory):
	
	if "_tokenized.txt" in file:
	
		#for each file create an absolute path to it
		filePath = os.path.join(directory, file)
		
		with open(filePath) as f:
			for line in f.readlines():
				line = line.lower().split()
				for word in line:
					words_all.append(word)
		
		f.close()

words_all_dict = defaultdict(int)
for word in words_all:
	words_all_dict[word] += 1

bag_of_words_5 = []
for key in words_all_dict:
	if words_all_dict[key] >= 5 and re.match('[a-z]', key):
		bag_of_words_5.append(key)

print(bag_of_words_5)

with open('bag_of_words_5.pickle', 'wb') as handle:
	pickle.dump(bag_of_words_5, handle, protocol=pickle.HIGHEST_PROTOCOL)		

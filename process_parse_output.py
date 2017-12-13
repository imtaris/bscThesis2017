#!/usr/bin/env python3

"""
File:	process_parse_output.py
Author:	Ivo Taris, s2188724
"""

import os
import sys
import re

#the absolute path to the original reviews
directory = '/home/s2188724/csicorpus/reviews_propname/'

for file in os.listdir(directory):
	
	if '_triples_with_frames.txt' in file:
	
		#for each file create an absolute path to it
		filePath = os.path.join(directory, file)
		
		#create a new filename for each file to be tokenized
		newFilePath = filePath[:-4] + '_processed' + filePath[-4:]
		nf = open(newFilePath,"w+")
		
		with open(filePath) as f:
			for line in f.readlines():
				
				#split on the "/" seperator
				fields = line.split('|')
				
				if len(fields) > 0 and fields[0] != 'top/top':

					lemma1 = fields[0].split('/')[0]
					lemma2 = fields[3].split('/')[0]
					postag1 = fields[1]
					postag2 = fields[4]
					relation = fields[2]
	
					nf.write(lemma1 + "\t" + postag2 + "\t" + relation + "\t" + lemma2 + "\t" + postag2 + "\n")
				
		nf.close()
		f.close()
	

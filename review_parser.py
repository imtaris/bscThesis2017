#!/usr/bin/env python3

"""
File:	review_parser.py
Author:	Ivo Taris, s2188724

This script takes the multi-line reviews of the CSI corpus as input,
processes each sentence using the Alpino dependency parser
parsing consists of using end hook 'triples_with_frames' 
and writes the resulting parse data to a new file.

Add in .bashrc using 'vim ~/.bashrc': 

export ALPINO_HOME=/net/aps/64/src/Alpino
export PATH=$PATH:$ALPINO_HOME/bin
Alpino end_hook=triples_with_frames -parse
"""

import os
import subprocess

#the absolute directory where the reviews are located
directory = '/home/s2188724/csicorpus/reviews/'

for file in os.listdir(directory):
	
	#use the tokenized file
	if '_tokenized' in file:
		
		#create source and target filenames
		filePath = os.path.join(directory, file)
		newFilePath = filePath[:-4] + '_triples_with_frames' + filePath[-4:]
		nf = open(newFilePath,"w+")

		with open(filePath) as f:
			print("\n" + filePath + "\n")
			for line in f.readlines():
				
				#visibility of processing
				print("\n" + line + "\n")
				
				#perform shell operation
				p1 = subprocess.Popen(["echo", line], stdout=subprocess.PIPE)
				p2 = subprocess.Popen(["sh", ".bashrc"], stdin=p1.stdout, stdout=subprocess.PIPE)
				p1.stdout.close()
				output,err = p2.communicate()
				nf.write(output.decode('UTF-8'))
				p2.stdout.close()
			nf.close()
		f.close()

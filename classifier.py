#!/usr/bin/env python3

"""
File:	classifier.py
Author:	Ivo Taris, s2188724
 
"""

from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold

import collections
from collections import Counter
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import numpy as np
import os
import re


def create_key(line, index1, index2):
	if 'punct' in line or '' in line :
		return str(None) + " " + str(None)
	key = line[index1] + " " + line[index2]
	if "\n" in key:
		# replaces multiple whitespace with single whitespace
		key = re.sub('\n', '', key)
	return key


def create_key_placeholder(line, placeholder_position, index):
	if 'punct' in line or '' in line:
		return None
	elif placeholder_position == 'left':
		key = "_ " + line[index]
	elif placeholder_position == 'right':
		key = line[index] + " _"
	if "\n" in key:
		# replaces multiple whitespace with single whitespace
		key = re.sub('\n', '', key)
	return key


def find_bigrams(input_list):
	bigram_list = []
	for i in range(len(input_list)-1):
		bigram_list.append((input_list[i] + " " + input_list[i+1]))
	return bigram_list


def create_features():

	# get the current working directory
	cwd = os.getcwd()

	# the absolute path to the totaltrainset
	directory = cwd + '/reviews_propname/alldata1/'	

	# a dictionary containing the features and label of each review
	csidata = {}
	csidata["features"] = []
	csidata["features_stopwords"] = []
	csidata["labels"] = []

	# process each file
	for file in os.listdir(directory):

		# use the tokenized reviews
		if "_tokenized.txt" in file:

			# for each file create an absolute path to it
			file_path = os.path.join(directory, file)

			file_features = defaultdict(int)
			file_features_stopwords = defaultdict(int)
			
			with open(file_path) as f:

				# identify the classlabel of the current file
				if 'Decep' in file_path:
					csidata["labels"].append(0)
				if 'Truth' in file_path:
					csidata["labels"].append(1)
				
				for line in f.readlines():
					line = line.lower().split()
						
					# F1: WORD UNIGRAMS
					
					for word in line:
						if re.match('[a-z]', word):
							file_features[word] += 1
							if word not in stopwords.words('dutch'):
								file_features_stopwords[word] += 1
					
					
					# F2: WORD BIGRAMS

					bigram_list = find_bigrams(line)
					for bigram in bigram_list:
						file_features[bigram] += 1
						if bigram.split()[0] not in stopwords.words('dutch') and bigram.split()[1] not in stopwords.words('dutch'):
							file_features_stopwords[bigram] += 1
					
			# use the matching Alpino output
			new_file_path = file_path[:-4] + '_triples_with_frames_processed.txt'
			

			# process each matching alpino parse output-file
			with open(new_file_path) as f:
				
				for line in f.readlines():
					line = line.lower().split("\t")

					# F3: STEM UNIGRAMS

					stem_pair = create_key(line, 0, 3).split()
					stem1 = stem_pair[0]
					stem2 = stem_pair[1]

					if stem1 is not 'None':
						file_features[stem1] += 1
						if line[0] not in stopwords.words('dutch'):
							file_features_stopwords[stem1] += 1
					
					if stem2 is not 'None':
						file_features[stem2] += 1
						if line[3] not in stopwords.words('dutch'):
							file_features_stopwords[stem2] += 1
					
					# F4: POS TAG UNIGRAMS

					postag_pair = create_key(line, 1, 4).split()
					postag1 = postag_pair[0]
					postag2 = postag_pair[1]
					
					if postag1 is not 'None':
						file_features[postag1] += 1
						if line[0] not in stopwords.words('dutch'):
							file_features_stopwords[postag1] += 1
					
					if postag2 is not 'None':
						file_features[postag2] += 1
						if line[3] not in stopwords.words('dutch'):
							file_features_stopwords[postag2] += 1

					# F5: DEPENDENCY RELATION UNIGRAMS

					dependency = line[2]
					file_features[dependency] = 1
					if line[0] not in stopwords.words('dutch') and line[3] not in stopwords.words('dutch'):
						file_features_stopwords[dependency] = 1

					# F6: STEM DEPENDENCY RELATION TRIPLE

					stem_dep_stem = create_key(line, 0, 3)
					if stem_dep_stem is not 'None None':
						file_features[stem_dep_stem] += 1
						if stem_dep_stem.split()[0] not in stopwords.words('dutch') and stem_dep_stem.split()[1] not in stopwords.words('dutch'):
							file_features_stopwords[stem_dep_stem] += 1

					# F7: POS TAG DEPENDENCY RELATION TRIPLE

					postag_dep_postag = create_key(line, 1, 4)
					if postag_dep_postag is not 'None None':
						file_features[postag_dep_postag] += 1
						if postag_dep_postag.split()[0] not in stopwords.words('dutch') and postag_dep_postag.split()[1] not in stopwords.words('dutch'):
							file_features_stopwords[postag_dep_postag] += 1

					# F8: STEM DEPENDENCY RELATION DOUBLE

					dep_stem = create_key(line, 2, 3)
					if dep_stem is not 'None None':
						file_features[dep_stem] += 1
						if dep_stem.split()[1] not in stopwords.words('dutch'):
							file_features_stopwords[dep_stem] += 1
					
					stem_dep = create_key(line, 0, 2)
					if stem_dep is not 'None None':
						file_features[stem_dep] += 1
						if stem_dep.split()[0] not in stopwords.words('dutch'):
							file_features_stopwords[stem_dep] += 1
					
					# F9: POS TAG DEPENDENCY RELATION DOUBLE

					dep_postag = create_key(line, 2, 4)
					if dep_postag is not 'None None':
						file_features[dep_postag] += 1
						if line[3] not in stopwords.words('dutch'):
							file_features_stopwords[dep_postag] += 1
					
					postag_dep = create_key(line, 1, 2)
					if postag_dep is not 'None None':
						file_features[postag_dep] += 1
						if line[0] not in stopwords.words('dutch'):
							file_features_stopwords[postag_dep] += 1

					# F10: STEM DEPENDENCY RELATION PLACEHOLDER DOUBLE

					placeholder_stem = create_key_placeholder(line, "left", 3)
					if placeholder_stem is not None:
						file_features[placeholder_stem] += 1
						if placeholder_stem.split()[1] not in stopwords.words('dutch'):
							file_features_stopwords[placeholder_stem] += 1		
					
					stem_placeholder = create_key_placeholder(line, "right", 0)
					if stem_placeholder is not None:
						file_features[stem_placeholder] += 1
						if stem_placeholder.split()[0] not in stopwords.words('dutch'):
							file_features_stopwords[stem_placeholder] += 1

					# F11: POS TAG DEPENDENCY RELATION PLACEHOLDER DOUBLE

					placeholder_postag = create_key_placeholder(line, "left", 4)
					if placeholder_postag is not None:
						file_features[placeholder_postag] += 1
						if line[3] not in stopwords.words('dutch'):
							file_features_stopwords[placeholder_postag] += 1		
					
					postag_placeholder = create_key_placeholder(line, "right", 1)
					if postag_placeholder is not None:
						file_features[postag_placeholder] += 1
						if line[0] not in stopwords.words('dutch'):
							file_features_stopwords[postag_placeholder] += 1
					
			f.close()
			csidata["features"].append(file_features)
			csidata["features_stopwords"].append(file_features_stopwords)
	return csidata


def main():

	# obtain the features of the reviews
	csidata = create_features()

	# convert the dictionaries to numpy arrays
	features = np.array(csidata["features"])
	features_stopwords = np.array(csidata["features_stopwords"])
	labels = np.array(csidata["labels"])

	# store the average result from each complete f-kold cross-validation
	av_accuracy = []
	av_accuracy_stopwords = []
	
	av_f1 = []
	av_f1_stopwords = []

	av_recall = []
	av_recall_stopwords = []
	
	av_precision = []
	av_precision_stopwords = []

	# repeat the k-fold cross-validation 20 times
	for i in range(20):

		kf = KFold(n_splits=10, shuffle=True)
		vectorizer = DictVectorizer()
		
		#classifier = svm.SVC(kernel='linear')
		classifier = LogisticRegression()

		# store the results of each k-fold iteration
		accuracy_kfold = []
		accuracy_stopwords_kfold = []
		f1_kfold = []
		f1_stopwords_kfold = []
		recall_kfold = []
		recall_stopwords_kfold = []
		precision_kfold = []
		precision_stopwords_kfold = []

		# perform 10-fold cross-validation
		for train_index, test_index in kf.split(features):
			
			X_train, X_test = features[train_index], features[test_index]
			X_train_stopwords, X_test_stopwords = features_stopwords[train_index], features_stopwords[test_index]
			label_train, label_test = labels[train_index], labels[test_index]

			# no filter
			X_train_ft = vectorizer.fit_transform(X_train, label_train)
			X_test_t = vectorizer.transform(X_test)
			classifier.fit(X_train_ft, label_train)
			predicted_y = classifier.predict(X_test_t)	
			accuracy_kfold.append(accuracy_score(label_test, predicted_y))
			f1_kfold.append(f1_score(label_test, predicted_y))
			recall_kfold.append(recall_score(label_test, predicted_y))
			precision_kfold.append(precision_score(label_test, predicted_y))

			# filtered for stopwords
			X_train_ft_stopwords = vectorizer.fit_transform(X_train_stopwords, label_train)
			X_test_t_stopwords = vectorizer.transform(X_test_stopwords)
			classifier.fit(X_train_ft_stopwords, label_train)
			predicted_y_stopwords = classifier.predict(X_test_t_stopwords)	
			accuracy_stopwords_kfold.append(accuracy_score(label_test, predicted_y_stopwords))
			f1_stopwords_kfold.append(f1_score(label_test, predicted_y_stopwords))
			recall_stopwords_kfold.append(recall_score(label_test, predicted_y_stopwords))
			precision_stopwords_kfold.append(precision_score(label_test, predicted_y_stopwords))

		# compute the average performance scores for each complete 10-fold cross validation
		av_accuracy_kfold = float(sum(accuracy_kfold))/max(len(accuracy_kfold),1)
		av_accuracy_stopwords_kfold = float(sum(accuracy_stopwords_kfold))/max(len(accuracy_stopwords_kfold),1)

		av_f1_kfold = float(sum(f1_kfold))/max(len(f1_kfold),1)
		av_f1_stopwords_kfold = float(sum(f1_stopwords_kfold))/max(len(f1_stopwords_kfold),1)

		av_recall_kfold = float(sum(recall_kfold))/max(len(recall_kfold),1)
		av_recall_stopwords_kfold = float(sum(recall_stopwords_kfold))/max(len(recall_stopwords_kfold),1)

		av_precision_kfold = float(sum(precision_kfold))/max(len(precision_kfold),1)
		av_precision_stopwords_kfold = float(sum(precision_stopwords_kfold))/max(len(precision_stopwords_kfold),1)

		# append each average performance score obtained from each complete 10-fold cross validation
		av_accuracy.append(av_accuracy_kfold)
		av_accuracy_stopwords.append(av_accuracy_stopwords_kfold)

		av_f1.append(av_f1_kfold)
		av_f1_stopwords.append(av_f1_stopwords_kfold)

		av_recall.append(av_recall_kfold)
		av_recall_stopwords.append(av_recall_stopwords_kfold)

		av_precision.append(av_precision_kfold)
		av_precision_stopwords.append(av_precision_stopwords_kfold)

	print("A: ", np.mean(av_accuracy),"R: ", np.mean(av_recall), "P: ", np.mean(av_precision), " F1: ", np.mean(av_f1))
	print("Stopwords. A: ", np.mean(av_accuracy_stopwords), "R: ", np.mean(av_recall_stopwords), "P: ", np.mean(av_precision_stopwords), " F1: ", np.mean(av_f1_stopwords))


if __name__ == '__main__':
	main()

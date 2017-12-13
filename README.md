# A lexico-syntactic approach to deception detection on Dutch reviews
Bachelor Thesis Information Science

Identifying deceptive writings on the web has become an important task. Although deception is often not identifiable by humans, classification systems show the ability to detect subtle differences between truthful and deceptive writings. As the amount of research on deception detection in Dutch is neglectible, this research focused on Dutch reviews from the CLiPS Stylometry Investigation (CSI) corpus. The results demonstrate that classifying the reviews as truthful or deceptive can be improved from 72.2% to 79.3% by incorporating both text-based lexical cues and syntactic patterns as features, and by filtering for stopwords. This repository provides all code and data needed to recreate the research.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See prerequisites for notes on how to deploy the project on a live system, and follow subsequent step to obtain the results.

### Prerequisites

To fully recreate the research requires:
* Python 3.x: https://www.python.org/downloads/
* A copy of the CSI corpus: https://www.clips.uantwerpen.be/datasets/csi-corpus
* A recent version of the Alpino parser: http://www.let.rug.nl/vannoord/alp/Alpino/
* The scikit-learn package: http://scikit-learn.org/stable/install.html
* The Natural Language Toolkit (NLTK) package: http://www.nltk.org/install.html
* Numpy
* Pickle

### CSI Corpus

The created classification system is trained and tested using the reviews from the CLiPS Stylometry Investigation (CSI) corpus, a Dutch corpus developed by the University of Antwerp. The corpus is updated on an annual basis and currently contains 1298 reviews and 517 essays written by 661 students. Whereas only the reviews are annotated as being truthful or deceptive, this research will focus only on the 1298 reviews written by 618 authors. 

The corpus contains the original reviews and a preproccesed version. In this preprocessed version, all the product names have been replaced with a *propname* tag. This version will be used to improve the robustness of the classifier with respect to cross-domain applications.

## Preprocess the reviews

The script 'preprocessor_propname.py' requires the absolute path to the 'reviews_propname' directory.

Once set, run 'python3 preprocessor_propname.py' to create a tokenized version of each review. The name of the new files ends with the string '_tokenized.txt'.

### Obtain the Alpino parse output

The script 'review_parser.py' requires the absolute path to the 'reviews_propname' directory.

Add in .bashrc using 'vim ~/.bashrc' the following three lines: 

export ALPINO_HOME=/net/aps/64/src/Alpino
export PATH=$PATH:$ALPINO_HOME/bin
Alpino end_hook=triples_with_frames -parse

Once set, run 'python3 review_parser.py' to obtain the Alpino parse output. This script takes the tokenized multi-line reviews of the CSI corpus as input and processes each sentence using the Alpino dependency parser. Parsing uses the end hook 'triples_with_frames' and writes the resulting parse data to a new file. The name of the new files ends with the string '_triples_with_frames.txt'.

### Process the Alpino parse output

The script 'process_parse_output.py' requires the absolute path to the 'reviews_propname' directory.

Once set, run 'python3 process_parse_output.py' to extract the five selected fields of each Alpino parse output line to a new file. The name of the new files ends with the string '_processed.txt'.

### Create features and perform classification

There are two classification files:
* classifier_chi2.py
* classifier.py

The former version has implemented a chi-square filter for selecting high informative features. The latter does not include such a filter. The file 'frequency_threshold_filter.py' can be used to create a pickle file that can be implemented by either of the classifiers to filter words that do not occur at least 5 times.

The function 'create_features()' in both classifiers produces all features for each review. Each feature can be commented out to customize the feature setup. In 'main()' one can choose which learning model is to be used.

## Acknowledgments

* Supervision by Dr. B. Plank, Assistant Professor, University of Groningen
* Data obtained from: Verhoeven, Ben & Daelemans Walter. (2014) CLiPS Stylometry Investigation (CSI) corpus: A Dutch corpus for the detection of age, gender, personality, sentiment and deception in text. In: Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC 2014). Reykjavik, Iceland.
* Used elements of the file 'featx.py' from Perkins, J. (2010). Python text processing with NLTK 2.0 cookbook. Packt Pub- lishing Ltd.

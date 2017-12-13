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



## Acknowledgments

* Supervision by Dr. B. Plank, Assistant Professor, University of Groningen
* Data obtained from: Verhoeven, Ben & Daelemans Walter. (2014) CLiPS Stylometry Investigation (CSI) corpus: A Dutch corpus for the detection of age, gender, personality, sentiment and deception in text. In: Proceedings of the 9th International Conference on Language Resources and Evaluation (LREC 2014). Reykjavik, Iceland.
* Used elements of the file 'featx.py' from Perkins, J. (2010). Python text processing with NLTK 2.0 cookbook. Packt Pub- lishing Ltd.

import imp
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Parse a file using a stored model.")
parser.add_argument("--features", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--sentences", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--model_path", help="Model path.", required=True)
parser.add_argument("--algorithm", help="Chosen algorithm.", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')
algorithm = imp.load_source('io', 'code/parsing/algorithms/'+args.algorithm+'.py')

features = io.read_features(args.features)
sentences = io.read_conll_sentences(args.sentences)
labels = [[token['dependency_graph'] for token in sentence] for sentence in sentences]

algorithm.fit(features, labels, model_path=args.model_path, save_every_iteration=True)

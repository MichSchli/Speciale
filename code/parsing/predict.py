import imp
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Parse a file using a stored model.")
parser.add_argument("--features", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--sentences", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (CoNLL format).", required=True)
parser.add_argument("--model_path", help="Model path.", required=True)
parser.add_argument("--algorithm", help="Output filepath (CoNLL format).", required=True)
parser.add_argument("--feature_mode", help="Feature set in use.", required=True)
args = parser.parse_args()

io = imp.load_source('io', 'code/common/io.py')
algorithm = imp.load_source('io', 'code/parsing/algorithms/'+args.algorithm+'.py')

features = {}

if args.feature_mode == 'sentence':
    features['sentence'] = io.read_sentence_features(args.features+'.sentence.feat')
elif args.feature_mode == 'character':
    features['character'] = io.read_character_features(args.features+'.character.feat')
elif args.feature_mode == 'both':
    features['sentence'] = io.read_sentence_features(args.features+'.sentence.feat')
    features['character'] = io.read_character_features(args.features+'.character.feat')

sentences = io.read_conll_sentences(args.sentences)

labels = algorithm.predict(features, model_path=args.model_path)

for sentence, label in zip(sentences, labels):
    for token, l in zip(sentence, label):
        token['dependency_graph'] = l

io.write_conll_sentences(sentences, args.outfile)

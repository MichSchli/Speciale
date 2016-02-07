import imp
import argparse
import numpy as np

io = imp.load_source('io', 'code/common/io.py')
embeddings = imp.load_source('embedding_reader', 'code/common/embedding_reader.py')

parser = argparse.ArgumentParser(description="Featurize a conll sentence file.")
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (Feature format).", required=True)
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)
features = []

embedding_model = embeddings.GloveReader()

def __featurize_token(token):
    global embedding_model

    feature = embedding_model[token['token'].lower()]
    return feature

for sentence in sentences:
    features.append([])
    for token in sentence:
        features[-1].append(__featurize_token(token))

io.write_features(features, args.outfile)

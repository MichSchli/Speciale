import imp
import argparse
import numpy as np

io = imp.load_source('io', 'code/common/io.py')

parser = argparse.ArgumentParser(description="Featurize a conll sentence file.")
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--outfile", help="Output filepath (Feature format).", required=True)
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)
features = []

def __featurize_token(token):
    return np.random.random_sample((5,))

for sentence in sentences:
    features.append([])
    for token in sentence:
        features[-1].append(__featurize_token(token))

io.write_features(features, args.outfile)

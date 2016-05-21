import imp
import argparse

io = imp.load_source('io', 'code/common/io.py')

parser = argparse.ArgumentParser(description="Do the projection from multiple probabilistic parses.")
parser.add_argument("--infile", help="Target language", required=True)
parser.add_argument("--outfile", help="Target language", required=True)
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)

for sentence in sentences:
    uniform = 1.0/len(sentence)
    
    for i,word in enumerate(sentence):
        word['dependency_graph'] = [uniform if i+1 != j else 0 for j in range(len(sentence)+1)]

io.write_conll_sentences(sentences, args.outfile)

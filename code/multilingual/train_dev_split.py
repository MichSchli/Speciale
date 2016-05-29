import imp
import argparse
import random

io = imp.load_source('io', 'code/common/io.py')
definitions = imp.load_source('definitions', 'code/common/definitions.py')
conll_column_headers = definitions.conll_column_headers

parser = argparse.ArgumentParser(description="Split a file in dev and train sets.")
parser.add_argument("--infile", help="Input filepath (CoNLL format).", required=True)
parser.add_argument("--train", help="Train filepath (CoNLL format).", required=True)
parser.add_argument("--dev", help="Dev filepath (CoNLL format).", required=True)
parser.add_argument("--devsize", help="Size of dev (float).", required=True, type=float)
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)
random.shuffle(sentences)
dev_sentence_count = int(len(sentences) * args.devsize)
print("Dev size: "+str(dev_sentence_count))

dev_sentences = sentences[:dev_sentence_count]
train_sentences = sentences[dev_sentence_count:]

io.write_conll_sentences(dev_sentences, args.dev)
io.write_conll_sentences(train_sentences, args.train)

import imp
import argparse

io = imp.load_source('io', 'code/common/io.py')

parser = argparse.ArgumentParser(description="Do the projection from multiple probabilistic parses.")
parser.add_argument("--infile", help="Source language", required=True)
args = parser.parse_args()

sentences = io.read_conll_sentences(args.infile)

new_sentences = []
for sentence in sentences:
    t_sum = 0
    for word in sentence:
        p_sum = sum(word['dependency_graph'])
        if p_sum > 0:
            word['dependency_graph'] = [x / p_sum for x in word['dependency_graph']]
            t_sum += p_sum

    if t_sum != 0:
        new_sentences.append(sentence)

io.write_conll_sentences(new_sentences, args.infile)

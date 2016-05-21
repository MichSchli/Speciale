import imp
import argparse

io = imp.load_source('io', 'code/common/io.py')

parser = argparse.ArgumentParser(description="Do the projection from multiple probabilistic parses.")
parser.add_argument("--infile", help="Source language", required=True)
parser.add_argument("--salign", help="Projections", required=True)
parser.add_argument("--walign", help="Projections", required=True)
parser.add_argument("--outfile", help="Target language", required=True)
args = parser.parse_args()

s_sentences = io.read_conll_sentences(args.infile)
t_sentences = io.read_conll_sentences(args.outfile)

for sentmatch, projsentence in zip(open(args.salign, 'r'), open(args.walign, 'r')):
    if not projsentence.strip():
        continue
    
    match_parts = sentmatch.strip().split('\t')
    s_sentence = s_sentences[int(match_parts[0])]
    t_sentence = t_sentences[int(match_parts[1])]
    
    pairdict = {}
    read_pair = True
    for item in projsentence.strip().split(' '):
        if read_pair:
            if not item.split('-')[0]:
                print(projsentence)
            pair = [int(n) for n in item.split('-')]
            read_pair = False
        else:
            read_pair = True
            print(pair)
            print(' '.join([word['token'] for word in s_sentence]))
            print(' '.join([word['token'] for word in t_sentence]))

            if pair[0] not in pairdict:
                pairdict[pair[0]] = {}
                
            pairdict[pair[0]][pair[1]] = float(item)

    for i,s_token in enumerate(s_sentence):
        if i in pairdict:
            for j, value in enumerate(s_token['dependency_graph']):
                if j in pairdict:
                    for i_t, j_t in zip(list(pairdict[i].keys()), list(pairdict[j].keys())):
                        t_sentence[i_t]['dependency_graph'][j_t] += pairdict[i][i_t]*pairdict[j][j_t]*value

        #print(s_token['token'])
        #print(t_sentence)
            
            
io.write_conll_sentences(t_sentences, args.outfile)

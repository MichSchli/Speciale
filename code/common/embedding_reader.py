import numpy as np
import sys
import pickle

class EmbeddingReader():
    
    def __getitem__(self, item):
        if item in self.d:
            return self.d[item]
        else:
            return np.zeros_like(self.d['word'])


class GloveReader(EmbeddingReader):

    glove_path = 'misc/glove.6B/glove.6B.50d.txt'
    d = {}
    
    def __init__(self):
        print("Loading GLOVE-dataset...", file=sys.stderr)
        for line in open(self.glove_path):
            parts = line.strip().split(' ')

            self.d[parts[0]] = np.array([float(x) for x in parts[1:]])

class PolyglotReader(EmbeddingReader):

    polyglot_path = 'misc/Polyglot/polyglot-en.pkl'
    d = {}

    def __init__(self):
        print("Loading polyglot...", file=sys.stderr)

        f = open(self.polyglot_path, 'rb')
        words, vecs = pickle.load(f, encoding='latin1')

        self.d = dict(zip(words, vecs))
        
if __name__ == '__main__':
    gr = PolyglotReader()
    print(gr['word'].shape)

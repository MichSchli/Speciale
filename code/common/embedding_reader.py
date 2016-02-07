import numpy as np
import sys

class GloveReader:

    glove_path = 'misc/glove.6B/glove.6B.50d.txt'
    d = {}
    
    def __init__(self):
        print("Loading GLOVE-dataset...", file=sys.stderr)
        for line in open(self.glove_path):
            parts = line.strip().split(' ')

            self.d[parts[0]] = np.array([float(x) for x in parts[1:]])

    def __getitem__(self, item):
        if item in self.d:
            return self.d[item]
        else:
            return np.zeros_like(self.d['man'])
        
if __name__ == '__main__':
    gr = GloveReader()
    print(gr['man'])

import numpy as np

def __read_dep_graph(string):
    parts = string.strip().split(' ')
    return np.array([float(x) for x in parts])

def __write_dep_graph(graph):
    return ' '.join([str(x) for x in graph])

'''
Defines the conll sentence file structure. 
Terminology is (header name, function to read element, function to write element).
'''
conll_column_headers = [('id', int, str),
                        ('token', str, str),
                        ('lemma', str, str),
                        ('X', str, str),
                        ('X', str, str),
                        ('part_of_speech', str, str),
                        ('X', str, str),
                        ('X', str, str),
                        ('Y', str, str),
                        ('dependency_head_id', int, str),
                        ('X', str, str),
                        ('dependency_head_relation', str, str),
                        ('X', str, str),
                        ('X', str, str),
                        ('dependency_graph', __read_dep_graph, __write_dep_graph)]

import theano
from theano import tensor as T
import numpy as np
import pickle

class RNN():

    '''
    Fields:
    '''

    W_output = None
    
    max_words = None
    input_dimension = 5
    
    '''
    Class methods:
    '''

    def __init(self, max_words):
        self.max_words = max_words

        self.W_forward_forget = np.random.rand(self.max_words, self.input_dimension + self.max_words + 1)
        self.W_backward_forget = np.random.rand(self.max_words, self.input_dimension + self.max_words + 1)        
        
        self.W_forward_input = np.random.rand(self.max_words, self.input_dimension + 1)
        self.W_backward_input = np.random.rand(self.max_words, self.input_dimension + 1)        

        self.W_forward_cell = np.random.rand(self.max_words, self.input_dimension + 1)
        self.W_backward_cell = np.random.rand(self.max_words, self.input_dimension + 1)        

        self.W_forward_output = np.random.rand(self.max_words, self.input_dimension + 1)
        self.W_backward_output = np.random.rand(self.max_words, self.input_dimension + 1)        
        
        self.W_output = np.random.rand(self.max_words, self.max_words, self.max_words +1)

    '''
    Theano functions:
    '''

    def __theano_lstm(self, x, h_prev, c_prev, W_forget, W_input, W_cell, W_output):
        input_vector = T.concatenate((x, h_prev, [1]))

        forget_gate = T.nnet.sigmoid(T.dot(W_forget, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(W_input, input_vector))
        candidate_vector = T.nnet.tanh(T.dot(W_candidate, input_vector))

        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(W_output, input_vector))
        h = output * T.nnet.tanh(cell_state)
        return h, cell_state
    
    def __get_theano_prediction(self, I, W_forward, W_backward, W_output):
        pass

def fit(sentence_instances, sentence_labels, model_path=None, save_every_iteration=False):
    pass

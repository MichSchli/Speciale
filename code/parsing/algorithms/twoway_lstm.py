import numpy as np

class RNN():

    '''
    Fields:
    '''

    hidden_dimension = 2
    input_dimension = 50

    '''
    Class methods:
    '''

    def __init__(self):
        self.W_output = np.random.rand(self.hidden_dimension)

        self.W_forward_forget = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)
        self.W_forward_input = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)
        self.W_forward_cell = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)
        self.W_forward_output = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)

        self.W_backward_forget = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)
        self.W_backward_input = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)
        self.W_backward_cell = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)
        self.W_backward_output = np.random.rand(self.hidden_dimension, self.input_dimension + self.hidden_dimension + 1)


    '''
    Theano functions:
    '''

    def __theano_lstm(self, x, h_prev, c_prev, W_forget, W_input, W_cell, W_output):

        input_vector = T.concatenate((x, h_prev, [1]))

        forget_gate = T.nnet.sigmoid(T.dot(W_forget, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(W_input, input_vector))
        candidate_vector = T.tanh(T.dot(W_cell, input_vector))
        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(W_output, input_vector))
        h = output * T.tanh(cell_state)
        return h, cell_state

def fit(features, labels, model_path=None, save_every_iteration=False):
    pass

def predict(sentence_list, model_path=None):
    predictions = []
    for sentence in sentence_list:
        predictions.append([])
        for token_feature in sentence:
            predictions[-1].append(np.random.uniform(0.0, 1.0, len(sentence)+1))

    return predictions

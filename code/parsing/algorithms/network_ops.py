from theano import tensor as T
import theano
import numpy as np

def single_lstm(x, h_prev, c_prev, W_forget, W_input, W_cell, W_output):
    #maybe add bias here instead of when making features
    input_vector = T.concatenate((x, h_prev))

    forget_gate = T.nnet.sigmoid(T.dot(W_forget, input_vector))
    input_gate = T.nnet.sigmoid(T.dot(W_input, input_vector))
    candidate_vector = T.tanh(T.dot(W_cell, input_vector))
    cell_state = forget_gate*c_prev + input_gate * candidate_vector

    output = T.nnet.sigmoid(T.dot(W_output, input_vector))
    h = output * T.tanh(cell_state)
    return h, cell_state

'''
def lstm_layer(Vs, W_forget, W_input, W_cell, W_output, forwards=True, hidden_dimension_size=None):
    h0 = np.zeros(hidden_dimension_size)
    c0 = np.zeros(hidden_dimension_size)

    lstm_preds, _ = theano.scan(fn=single_lstm,
                        outputs_info=[h0,c0],
                        sequences=Vs,
                        non_sequences=[W_forget,
                                       W_input,
                                       W_cell,
                                       W_output],
                        go_backwards=not forwards)

    # Discard the cell values:
    return lstm_preds[0]
'''

class lstm_layer():

    def __init__(self, hidden_neurons, direction):
        self.hidden_neurons=hidden_neurons
        self.direction=direction

    def function(self, Vs, W_forget, W_input, W_cell, W_output):
        h0 = T.zeros(self.hidden_neurons)
        c0 = T.zeros(self.hidden_neurons)

        lstm_preds, _ = theano.scan(fn=single_lstm,
                        outputs_info=[h0,c0],
                        sequences=Vs,
                        non_sequences=[W_forget,
                                       W_input,
                                       W_cell,
                                       W_output],
                                    go_backwards=not self.direction)

        # Discard the cell values:
        return lstm_preds[0]


class bidirectional_lstm_layer():

    def __init__(self, hidden_neurons):
        self.hidden_neurons=hidden_neurons

        self.forward = lstm_layer(hidden_neurons, True)
        self.backward = lstm_layer(hidden_neurons, False)
    
    def function(self, Vs, W_forget, W_input, W_cell, W_output):
        
        forwards_h = self.forward.function(Vs, W_forget[0], W_input[0], W_cell[0], W_output[0])
        backwards_h = self.backward.function(Vs, W_forget[1], W_input[1], W_cell[1], W_output[1])

        return T.concatenate((forwards_h, backwards_h), axis=1)

    
class fourdirectional_lstm_layer():
    
    def __init__(self, hidden_neurons):
        self.hidden_neurons = hidden_neurons

        self.layer = bidirectional_lstm_layer(hidden_neurons)
    
    def function(self, VM, W_forget, W_input, W_cell, W_output):
        lstm_sidewards, _ = theano.scan(fn=self.layer.function,
                                        outputs_info=None,
                                        sequences=VM,
                                        non_sequences=[W_forget[:2], W_input[:2], W_cell[:2], W_output[:2]])

        transpose_vm = VM.transpose(1,0,2)

        lstm_downwards, _ = theano.scan(fn=self.layer.function,
                                        outputs_info=None,
                                        sequences=transpose_vm,
                                        non_sequences=[W_forget[2:], W_input[2:], W_cell[2:], W_output[2:]])

        return T.concatenate((lstm_sidewards,lstm_downwards.transpose(1,0,2)), axis=2)

        
class corner_lstm_layer():

    
    def __init__(self, hidden_neurons, x_length, y_length):
        self.hidden_neurons = hidden_neurons
        self.x_length = x_length
        self.y_length = y_length

        self.layer = bidirectional_lstm_layer(hidden_neurons)

    def __wrapper(Vs, prev_hs, W_forget, W_input, W_cell, W_output):
        inputs = T.concatenate((Vs, prev_hs), axis=1)

        return self.layer.function(inputs, W_forget, W_input, W_cell, W_output)
        
    def function(self, VM, W_forget, W_input, W_cell, W_output):
        init_hs = T.zeros((Vm.shape[0], self.hidden_neurons))
        lstm_downwards, _ = theano.scan(fn=self.__wrapper,
                                        outputs_info=init_hs,
                                        sequences=pairwise_vs,
                                        non_sequences=[W_forget[:2], W_input[:2], W_cell[:2], W_output[:2]])

        lstm_upwards, _ = theano.scan(fn=self.__wrapper,
                                        outputs_info=init_hs,
                                        sequences=pairwise_vs,
                                        non_sequences=[W_forget[2:], W_input[2:], W_cell[2:], W_output[2:]],
                                        go_backwards=True)

        return T.concatenate((lstm_downwards, lstm_upwards), axis=2)



'''
def bidirectional_lstm_layer(Vs, W_forget, W_input, W_cell, W_output, hidden_dimension_size=None):

    forwards_h = lstm_layer(Vs, W_forget[0], W_input[0], W_cell[0], W_output[0], forwards=True, hidden_dimension_size=hidden_dimension_size)
    backwards_h = lstm_layer(Vs, W_forget[1], W_input[1], W_cell[1], W_output[1], forwards=False, hidden_dimension_size=hidden_dimension_size)

    return T.concatenate((forwards_h, backwards_h), axis=1)

def fourdirectional_lstm_layer(VM, W_forget, W_input, W_cell, W_output, hidden_dimension_size=None):
    lstm_sidewards, _ = theano.scan(fn=lambda a,b,c,d,e: bidirectional_lstm_layer(a,b,c,d,e, hidden_dimension_size=self.hidden_dimension),
                                                 outputs_info=None,
                                                 sequences=pairwise_vs,
                                                 non_sequences=[W_forget[:2], W_input[:2], W_cell[:2], W_output[:2]])

    transpose_vs = pairwise_vs.transpose(1,0,2)

    lstm_downwards, _ = theano.scan(fn=lambda a,b,c,d,e: bidirectional_lstm_layer(a,b,c,d,e, hidden_dimension_size=self.hidden_dimension),
                                                 outputs_info=None,
                                                 sequences=transpose_vs,
                                                 non_sequences=[W_forget[2:], W_input[2:], W_cell[2:], W_output[2:]])

    return T.concatenate((lstm_sidewards,lstm_downwards.transpose(1,0,2)), axis=2)
'''

def linear_layer(input_vector, weight_matrix):
    input_with_bias = T.concatenate((input_vector, [1]))
    return T.dot(weight_matrix, input_with_bias)

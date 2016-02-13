from theano import tensor as T
import theano
import numpy as np


class single_lstm(): 
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons

        self.name = name

        #Initialize theano variables:
        self.W_forget_theano = T.dmatrix(self.name + '_forget_weight')
        self.W_input_theano = T.dmatrix(self.name + '_input_weight')
        self.W_candidate_theano = T.dmatrix(self.name + '_candidate_weight')
        self.W_output_theano = T.dmatrix(self.name + '_output_weight')

        #Initialize python variables:
        self.W_forget = np.random.rand(self.output_neurons, self.input_neurons + self.output_neurons + 1)
        self.W_input = np.random.rand(self.output_neurons, self.input_neurons + self.output_neurons + 1)
        self.W_candidate = np.random.rand(self.output_neurons, self.input_neurons + self.output_neurons + 1)
        self.W_output = np.random.rand(self.output_neurons, self.input_neurons + self.output_neurons + 1)


    def get_theano_weights(self):
        return self.W_forget_theano, self.W_input_theano, self.W_candidate_theano, self.W_output_theano

    def get_python_weights(self):
        return self.W_forget, self.W_input, self.W_candidate, self.W_output
    
    
    def function(self, x, h_prev, c_prev):
        input_vector = T.concatenate((x, h_prev, [1]))

        forget_gate = T.nnet.sigmoid(T.dot(self.W_forget_theano, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(self.W_input_theano, input_vector))
        candidate_vector = T.tanh(T.dot(self.W_candidate_theano, input_vector))
        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(self.W_output_theano, input_vector))
        h = output * T.tanh(cell_state)
        return h, cell_state
    

class lstm_layer():
    
    def __init__(self, name, input_neurons, output_neurons, direction):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons
        self.direction=direction

        self.name = name
        self.neuron=single_lstm(name, input_neurons, output_neurons)

    def get_theano_weights(self):
        return self.neuron.get_theano_weights()

    def get_python_weights(self):
        return self.neuron.get_python_weights()
        
    def function(self, Vs):
        h0 = T.zeros(self.output_neurons)
        c0 = T.zeros(self.output_neurons)

        lstm_preds, _ = theano.scan(fn=self.neuron.function,
                        outputs_info=[h0,c0],
                        sequences=Vs,
                        non_sequences=None,
                        go_backwards=not self.direction)

        # Discard the cell values:
        return lstm_preds[0]


class bidirectional_lstm_layer():

    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.forward = lstm_layer(name + '_forward', input_neurons, output_neurons, True)
        self.backward = lstm_layer(name + '_backward', input_neurons, output_neurons, False)

    def get_theano_weights(self):
        return self.forward.get_theano_weights() + self.backward.get_theano_weights()

    def get_python_weights(self):
        return self.forward.get_python_weights() + self.backward.get_python_weights()
        
    
    def function(self, Vs):
        
        forwards_h = self.forward.function(Vs)
        backwards_h = self.backward.function(Vs)

        return T.concatenate((forwards_h, backwards_h), axis=1)

    
class fourdirectional_lstm_layer():
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.sideward_layer = bidirectional_lstm_layer(name + '_sideward', input_neurons, output_neurons)
        self.downward_layer = bidirectional_lstm_layer(name + '_downward', input_neurons, output_neurons)

    def get_theano_weights(self):
        return self.sideward_layer.get_theano_weights() + self.downward_layer.get_theano_weights()

    def get_python_weights(self):
        return self.sideward_layer.get_python_weights() + self.downward_layer.get_python_weights()
        
    
    def function(self, VM):
        lstm_sidewards, _ = theano.scan(fn=self.sideward_layer.function,
                                        outputs_info=None,
                                        sequences=VM,
                                        non_sequences=None)

        transpose_vm = VM.transpose(1,0,2)

        lstm_downwards, _ = theano.scan(fn=self.downward_layer.function,
                                        outputs_info=None,
                                        sequences=transpose_vm,
                                        non_sequences=None)

        return T.concatenate((lstm_sidewards,lstm_downwards.transpose(1,0,2)), axis=2)

#Fix later:        
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
                                        sequences=VM,
                                        non_sequences=[W_forget[:2], W_input[:2], W_cell[:2], W_output[:2]])

        lstm_upwards, _ = theano.scan(fn=self.__wrapper,
                                        outputs_info=init_hs,
                                        sequences=VM,
                                        non_sequences=[W_forget[2:], W_input[2:], W_cell[2:], W_output[2:]],
                                        go_backwards=True)

        return T.concatenate((lstm_downwards, lstm_upwards), axis=2)

    
class linear_layer():

    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.name = name
        
        if output_neurons == 1:
            self.weight_matrix_theano = T.dvector(name + '_weight')
            self.weight_matrix = np.random.rand(self.input_neurons+1)
        else:
            self.weight_matrix_theano = T.dmatrix(name + '_weight')
            self.weight_matrix = np.random.rand(self.output_neurons, self.input_neurons+1)

            
    def get_theano_weights(self):
        return (self.weight_matrix_theano,)

    def get_python_weights(self):
        return (self.weight_matrix,)
        
    def function(self, input_vector):
        input_with_bias = T.concatenate((input_vector, [1]))
        return T.dot(self.weight_matrix_theano, input_with_bias)


class linear_tensor_convolution_layer():

    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.name = name
        
        self.neuron = linear_layer(name, input_neurons, output_neurons)

    def get_theano_weights(self):
        return self.neuron.get_theano_weights()

    def get_python_weights(self):
        return self.neuron.get_python_weights()

    def function(self, input_tensor):
        m_x = input_tensor.shape[0]
        m_y = input_tensor.shape[1]
        flattened = T.reshape(input_tensor, newshape=(m_x*m_y, input_tensor.shape[2]))
        
        outputs, _ = theano.scan(fn=self.neuron.function,
                                 sequences=flattened,
                                 non_sequences=None)

        if self.output_neurons == 1:
            return T.reshape(outputs, newshape=(m_x,m_y))
        else:
            return T.reshape(outputs, newshape=(m_x,m_y,self.output_neurons))


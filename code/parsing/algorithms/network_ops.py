from theano import tensor as T
import theano
import numpy as np

class single_lstm(): 

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons

        self.name = name

        #Initialize theano variables:
        self.W_forget_theano = T.fmatrix(self.name + '_forget_weight')
        self.W_input_theano = T.fmatrix(self.name + '_input_weight')
        self.W_candidate_theano = T.fmatrix(self.name + '_candidate_weight')
        self.W_output_theano = T.fmatrix(self.name + '_output_weight')

        #Initialize python variables:

        high_init = np.sqrt(6)/np.sqrt(self.input_neurons + 2*self.output_neurons)
        low_init = -high_init
        
        s = (self.output_neurons, self.input_neurons + self.output_neurons + 1)
        self.W_forget = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)
        self.W_input = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)
        self.W_candidate = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)
        self.W_output = np.random.uniform(low=low_init, high=high_init, size=s).astype(np.float32)

        #Initialize forget bias to one:
        self.W_forget[-1] = np.ones_like(self.W_forget[-1], dtype=np.float32)

    def set_training(self, training):
        self.training=training

    def update_weights(self, update_list):
        self.W_forget = update_list[0]
        self.W_input = update_list[1]
        self.W_candidate = update_list[2]
        self.W_output = update_list[3]

    def weight_count(self):
        return 4
        
    def get_theano_weights(self):
        return self.W_forget_theano, self.W_input_theano, self.W_candidate_theano, self.W_output_theano

    def get_python_weights(self):
        return self.W_forget, self.W_input, self.W_candidate, self.W_output
    
    
    def function(self, x, h_prev, c_prev):
        input_vector = T.concatenate((x, h_prev, T.ones(1)))
        
        forget_gate = T.nnet.sigmoid(T.dot(self.W_forget_theano, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(self.W_input_theano, input_vector))
        candidate_vector = T.tanh(T.dot(self.W_candidate_theano, input_vector))
        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(self.W_output_theano, input_vector))
        h = output * T.tanh(cell_state)
        return h, cell_state

class twodim_lstm(): 

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons

        self.name = name

        #Initialize theano variables:
        self.W_forget_theano_1 = T.fmatrix(self.name + '_forget_weight_1')
        self.W_forget_theano_2 = T.fmatrix(self.name + '_forget_weight_2')
        self.W_input_theano = T.fmatrix(self.name + '_input_weight')
        self.W_candidate_theano = T.fmatrix(self.name + '_candidate_weight')
        self.W_output_theano = T.fmatrix(self.name + '_output_weight')

        #Initialize python variables:

        high_init = np.sqrt(6)/np.sqrt(self.input_neurons + 2*self.output_neurons)
        low_init = -high_init
        
        s = (self.output_neurons, self.input_neurons + self.output_neurons*2 + 1)
        
        self.W_forget_1 = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_forget_2 = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_input = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_candidate = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_output = np.random.uniform(low=low_init, high=high_init, size=s)

        #Initialize forget bias to one:
        self.W_forget_1[-1] = np.ones_like(self.W_forget_1[-1])
        self.W_forget_2[-1] = np.ones_like(self.W_forget_2[-1])

    def set_training(self, training):
        self.training=training

    def update_weights(self, update_list):
        self.W_forget_1 = update_list[0]
        self.W_forget_2 = update_list[1]
        self.W_input = update_list[2]
        self.W_candidate = update_list[3]
        self.W_output = update_list[4]

    def weight_count(self):
        return 5
        
    def get_theano_weights(self):
        return self.W_forget_theano_1, self.W_forget_theano_2, self.W_input_theano, self.W_candidate_theano, self.W_output_theano

    def get_python_weights(self):
        return self.W_forget_1, self.W_forget_2, self.W_input, self.W_candidate, self.W_output
    
    
    def function(self, x, h_prev_1, c_prev_1, h_prev_2, c_prev_2):
        input_vector = T.concatenate((x, h_prev_1, h_prev_2, T.ones(1)))

        forget_gate_1 = T.nnet.sigmoid(T.dot(self.W_forget_theano_1, input_vector))
        forget_gate_2 = T.nnet.sigmoid(T.dot(self.W_forget_theano_2, input_vector))
        
        input_gate = T.nnet.sigmoid(T.dot(self.W_input_theano, input_vector))
        candidate_vector = T.tanh(T.dot(self.W_candidate_theano, input_vector))

        cell_state_1 = forget_gate_1*c_prev_1 + input_gate * candidate_vector
        cell_state_2 = forget_gate_2*c_prev_2 + input_gate * candidate_vector

        cell_state_total = cell_state_1 + cell_state_2

        output = T.nnet.sigmoid(T.dot(self.W_output_theano, input_vector))
        h = output * T.tanh(cell_state_total)
        return h, cell_state_total

    
class multilayer_lstm():

    def __init__(self, name,
                 input_bottom_neurons,
                 input_top_neurons,
                 output_bottom_neurons,
                 output_top_neurons,
                 run_forward):
        
        self.input_bottom_neurons = input_bottom_neurons
        self.input_top_neurons = input_top_neurons
        self.output_bottom_neurons = output_bottom_neurons
        self.output_top_neurons = output_top_neurons
        self.run_forward = run_forward
        self.name = name

        self.bottom_neuron=single_lstm(name, input_bottom_neurons + output_top_neurons, output_bottom_neurons)
        self.top_neuron=single_lstm(name, input_top_neurons + output_bottom_neurons, output_top_neurons)

    
    def set_training(self, training):
        self.training=training
        self.bottom_neuron.set_training(training)
        self.top_neuron.set_training(training)
        
    def update_weights(self, update_list):
        self.bottom_neuron.update_weights(update_list[:self.bottom_neuron.weight_count()])
        self.top_neuron.update_weights(update_list[self.bottom_neuron.weight_count():])

    def weight_count(self):
        return self.bottom_neuron.weight_count() + self.top_neuron.weight_count()
        
    def get_theano_weights(self):
        return self.bottom_neuron.get_theano_weights() + self.top_neuron.get_theano_weights()

    def get_python_weights(self):
        return self.bottom_neuron.get_python_weights() + self.top_neuron.get_python_weights()

    def call_bottom_neuron(self, BottomFeatureVector, PrevBottomH, PrevBottomC, PrevTopOutputVector):
        Joint = T.concatenate((PrevTopOutputVector, BottomFeatureVector))
        return self.bottom_neuron.function(Joint, PrevBottomH, PrevBottomC)
    
    def iterate_bottom(self, PrevTopOutputVector, BottomFeatureMatrix, BottomSequenceLength):
                
        BottomFeatureMatrix = BottomFeatureMatrix[:BottomSequenceLength]

        h0_bottom = T.zeros(self.output_bottom_neurons)
        c0_bottom = T.zeros(self.output_bottom_neurons)

        lstm_preds, _ = theano.scan(fn=self.call_bottom_neuron,
                        sequences=BottomFeatureMatrix,
                        outputs_info=[h0_bottom,c0_bottom],
                        non_sequences=PrevTopOutputVector,
                                    go_backwards=not self.run_forward)

        # Discard everything but the last hidden values:
        return lstm_preds[0][-1]

    def call_top_neuron(self, BottomFeatureMatrix, BottomSequenceLength, TopFeatureVector, PrevTopH, PrevTopC):
        PrevBottomOutputVector = self.iterate_bottom(PrevTopH, BottomFeatureMatrix, BottomSequenceLength)
        Joint = T.concatenate((PrevBottomOutputVector, TopFeatureVector))
        return self.top_neuron.function(Joint, PrevTopH, PrevTopC)    
    
    def function(self, TopFeatureMatrix, BottomFeatureTensor, BottomSequenceVector):
        h0_top = T.zeros(self.output_top_neurons)
        c0_top = T.zeros(self.output_top_neurons)

        lstm_preds, _ = theano.scan(fn=self.call_top_neuron,
                        sequences=[BottomFeatureTensor, BottomSequenceVector, TopFeatureMatrix],
                        outputs_info=[h0_top,c0_top],
                        non_sequences=None,
                                    go_backwards=not self.run_forward)

        # Discard the cell values:
        if self.run_forward:
            return lstm_preds[0]
        else:
            return lstm_preds[0][::-1]
    
    
    
class lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons, direction, get_cell_values=False):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons
        self.direction=direction

        self.name = name
        self.neuron=single_lstm(name, input_neurons, output_neurons)

        self.get_cell_values=get_cell_values
    
    def set_training(self, training):
        self.training=training
        self.neuron.set_training(training)
        
    def update_weights(self, update_list):
        self.neuron.update_weights(update_list)

    def weight_count(self):
        return self.neuron.weight_count()
    
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

        if self.direction:
            if not self.get_cell_values:
                return lstm_preds[0]
            else:
                return lstm_preds
        else:
            if not self.get_cell_values:
                return lstm_preds[0][::-1]
            else:
                return [lstm_preds[0][::-1], lstm_preds[1][::-1]]

            
class bidirectional_rnn_lstm():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.forward = lstm_layer(name + '_forward', input_neurons, output_neurons, True)
        self.backward = lstm_layer(name + '_backward', input_neurons, output_neurons, False)
    
    def set_training(self, training):
        self.training=training
        self.forward.set_training(training)
        self.backward.set_training(training)
        
    def update_weights(self, update_list):
        self.forward.update_weights(update_list[:self.forward.weight_count()])
        self.backward.update_weights(update_list[self.forward.weight_count():])

    def weight_count(self):
        return self.forward.weight_count() + self.backward.weight_count()
        
    def get_theano_weights(self):
        return self.forward.get_theano_weights() + self.backward.get_theano_weights()

    def get_python_weights(self):
        return self.forward.get_python_weights() + self.backward.get_python_weights()
        
    
    def function(self, Vs):
        
        forwards_h = self.forward.function(Vs)[-1]
        backwards_h = self.backward.function(Vs)[-1]

        return T.concatenate((forwards_h, backwards_h))


class bidirectional_lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.forward = lstm_layer(name + '_forward', input_neurons, output_neurons, True)
        self.backward = lstm_layer(name + '_backward', input_neurons, output_neurons, False)
    
    def set_training(self, training):
        self.training=training
        self.forward.set_training(training)
        self.backward.set_training(training)
        
    def update_weights(self, update_list):
        self.forward.update_weights(update_list[:self.forward.weight_count()])
        self.backward.update_weights(update_list[self.forward.weight_count():])

    def weight_count(self):
        return self.forward.weight_count() + self.backward.weight_count()
        
    def get_theano_weights(self):
        return self.forward.get_theano_weights() + self.backward.get_theano_weights()

    def get_python_weights(self):
        return self.forward.get_python_weights() + self.backward.get_python_weights()
        
    
    def function(self, Vs):
        
        forwards_h = self.forward.function(Vs)
        backwards_h = self.backward.function(Vs)

        return T.concatenate((forwards_h, backwards_h), axis=1)

    
class fourdirectional_lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.sideward_layer = bidirectional_lstm_layer(name + '_sideward', input_neurons, output_neurons)
        self.downward_layer = bidirectional_lstm_layer(name + '_downward', input_neurons, output_neurons)

    
    def set_training(self, training):
        self.training=training
        self.sideward_layer.set_training(training)
        self.downward_layer.set_training(training)
        

    def update_weights(self, update_list):
        self.sideward_layer.update_weights(update_list[:self.sideward_layer.weight_count()])
        self.downward_layer.update_weights(update_list[self.downward_layer.weight_count():])

    def weight_count(self):
        return self.sideward_layer.weight_count() + self.downward_layer.weight_count()
        
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

    
class corner_lstm_layer():
    
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.name = name
        self.d_forward = twodim_lstm(name + 'down_forward', input_neurons, output_neurons)
        self.d_backward = twodim_lstm(name + 'down_backward', input_neurons, output_neurons)
        self.u_forward = twodim_lstm(name + 'up_forward', input_neurons, output_neurons)
        self.u_backward = twodim_lstm(name + 'up_backward', input_neurons, output_neurons)

        self.lstms = [self.d_forward, self.d_backward, self.u_forward, self.u_backward]


    def update_weights(self, update_list):
        prev = 0
        for lstm in self.lstms:
            cur = prev + lstm.weight_count()
            lstm.update_weights(update_list[prev:cur])
            prev = cur

    def set_training(self, training):
        self.training=training
        self.d_forward.set_training(training)
        self.d_backward.set_training(training)
        self.u_forward.set_training(training)
        self.u_backward.set_training(training)
                    
    def weight_count(self):
        return sum([lstm.weight_count() for lstm in self.lstms])
        
    def get_theano_weights(self):
        return tuple(w for lstm in self.lstms for w in lstm.get_theano_weights())
    
    def get_python_weights(self):
        return tuple(w for lstm in self.lstms for w in lstm.get_python_weights())

    def __lstm_wrapper(self, input_matrix, prev_hidden_matrix, prev_cell_matrix, lstm, go_forwards=True):
        h0 = T.zeros(self.output_neurons)
        c0 = T.zeros(self.output_neurons)

        lstm_preds, _ = theano.scan(fn=lstm.function,
                        outputs_info=[h0,c0],
                        sequences=[input_matrix, prev_hidden_matrix, prev_cell_matrix],
                        non_sequences=None,
                                    go_backwards=not go_forwards)

        if go_forwards:
            return lstm_preds
        else:
            return lstm_preds[0][::-1], lstm_preds[1][::-1]
    
    def function(self, input_tensor):
        init_hs = T.zeros((input_tensor.shape[1], self.output_neurons))
        init_cs = T.zeros((input_tensor.shape[1], self.output_neurons))

        lstm_out_1, _ = theano.scan(fn=lambda a,b,c: self.__lstm_wrapper(a,b,c,self.d_forward, go_forwards=True),
                                      outputs_info=[init_hs,init_cs],
                                      sequences=input_tensor,
                                      non_sequences=None)
        
        lstm_out_2, _ = theano.scan(fn=lambda a,b,c: self.__lstm_wrapper(a,b,c,self.d_backward, go_forwards=False),
                                      outputs_info=[init_hs,init_cs],
                                      sequences=input_tensor,
                                      non_sequences=None)
        
        lstm_out_3, _ = theano.scan(fn=lambda a,b,c: self.__lstm_wrapper(a,b,c,self.u_forward, go_forwards=True),
                                      outputs_info=[init_hs,init_cs],
                                      sequences=input_tensor,
                                      non_sequences=None,
                                      go_backwards=True)

        lstm_out_4, _ = theano.scan(fn=lambda a,b,c: self.__lstm_wrapper(a,b,c,self.u_backward, go_forwards=False),
                                      outputs_info=[init_hs,init_cs],
                                      sequences=input_tensor,
                                      non_sequences=None,
                                      go_backwards=True)


        return T.concatenate((lstm_out_1[0],
                              lstm_out_2[0],
                              lstm_out_3[0][::-1],
                              lstm_out_4[0][::-1]), axis=2)

    
class linear_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.name = name

        high_init = np.sqrt(6)/np.sqrt(input_neurons + output_neurons)
        low_init = -np.sqrt(6)/np.sqrt(input_neurons + output_neurons)
        
        if output_neurons == 1:
            self.weight_matrix_theano = T.fvector(name + '_weight')
            self.weight_matrix = np.random.uniform(low=low_init, high=high_init, size=self.input_neurons+1).astype(np.float32)
        else:
            self.weight_matrix_theano = T.fmatrix(name + '_weight')
            self.weight_matrix = np.random.uniform(low=low_init, high=high_init, size=(self.output_neurons, self.input_neurons+1)).astype(np.float32)

    def set_training(self, training):
        self.training=training
    
    def update_weights(self, update_list):
        self.weight_matrix = update_list[0]
        
    def weight_count(self):
        return 1
            
    def get_theano_weights(self):
        return (self.weight_matrix_theano,)

    def get_python_weights(self):
        return (self.weight_matrix,)
        
    def function(self, input_vector):
        input_with_bias = T.concatenate((input_vector, T.ones(1)))
        return T.dot(self.weight_matrix_theano, input_with_bias)


class linear_tensor_convolution_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.name = name
        
        self.neuron = linear_layer(name, input_neurons, output_neurons)

    def set_training(self, training):
        self.training=training
        self.neuron.set_training(training)
        
    def update_weights(self, update_list):
        self.neuron.update_weights(update_list)
        
    def weight_count(self):
        return self.neuron.weight_count()
        
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


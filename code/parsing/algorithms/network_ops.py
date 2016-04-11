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
        self.W_forget_theano = T.dmatrix(self.name + '_forget_weight')
        self.W_input_theano = T.dmatrix(self.name + '_input_weight')
        self.W_candidate_theano = T.dmatrix(self.name + '_candidate_weight')
        self.W_output_theano = T.dmatrix(self.name + '_output_weight')

        #Initialize python variables:

        high_init = np.sqrt(6)/np.sqrt(self.input_neurons + 2*self.output_neurons)
        low_init = -high_init
        
        s = (self.output_neurons, self.input_neurons + self.output_neurons + 1)
        self.W_forget = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_input = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_candidate = np.random.uniform(low=low_init, high=high_init, size=s)
        self.W_output = np.random.uniform(low=low_init, high=high_init, size=s)

        #Initialize forget bias to one:
        self.W_forget[-1] = np.ones_like(self.W_forget[-1])

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
        input_vector = T.concatenate((x, h_prev, [1]))
        
        forget_gate = T.nnet.sigmoid(T.dot(self.W_forget_theano, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(self.W_input_theano, input_vector))
        candidate_vector = T.tanh(T.dot(self.W_candidate_theano, input_vector))
        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(self.W_output_theano, input_vector))
        h = output * T.tanh(cell_state)
        return h, cell_state
    
    
class threed_grid_lstm_cell():

    
    def __init__(self, name, input_shapes, output_shapes):
        assert(len(input_shapes) == 3)
        assert(len(output_shapes) == 3)
        
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.name = name

        H_shape = sum(self.input_shapes)
        
        self.neurons = [None]*len(input_shapes)
        for i, output_neurons in enumerate(output_shapes):
            self.neurons[i] = single_lstm(name + '_' + str(i), H_shape, output_neurons)
    
    def update_weights(self, update_list):
        counter = 0
        for neuron in self.neurons:
            neuron.update_weights(update_list[counter:counter+neuron.weight_count()])
            counter += neuron.weight_count()

            
    def weight_count(self):
        return sum([neuron.weight_count() for neuron in self.neurons])
        
    def get_theano_weights(self):
        w = fisk
        for neuron in self.neurons:
            w += neuron.get_theano_weights()
            
    def get_python_weights(self):
        w = fisk
        for neuron in self.neurons:
            w += neuron.get_python_weights()
        
    def function(self, x, H, C):
        h0, c0 = self.neurons[0].function(x, H, C[0])
        h1, c1 = self.neurons[1].function(x, H, C[1])
        h2, c2 = self.neurons[2].function(x, H, C[2])

        new_hs = [h0, h1, h2]
        new_cs = [c0, c1, c2]
        
        return new_hs, new_cs


class threed_grid_lstm_block():

    def __init__(self, name, input_shapes, output_shapes):
        assert(input_shapes[1] == output_shapes[1])
        assert(input_shapes[2] == output_shapes[2])
        
        self.name = name
        self.neuron = threed_grid_lstm_cell(name, input_shapes, output_shapes)

    def update_weights(self, update_list):
        self.neuron.update_weights(update_list)

    def weight_count(self):
        return self.neuron.weight_count()
    
    def get_theano_weights(self):
        return self.neuron.get_theano_weights()

    def get_python_weights(self):
        return self.neuron.get_python_weights()
        
    def function(self, input_tensor):
        def ev(h_below, c_below, h_y_axis, c_y_axis, h_x_axis, c_x_axis):
            hs, cs = self.neuron.function(h_below, T.conc((h_y_axis, h_x_axis)), (c_below, c_y_axis, c_x_axis))

            return hs[0], cs[0], hs[1], cs[1], hs[2], cs[2]

        def ev_row(hs_prevlayer, cs_prevlayer, hs_prevrow, cs_prevrow):
            row_init_h = T.zeros(neurons_per_downward_cell) # Number of neurons per cell in the column direction
            row_init_c = T.zeros(neurons_per_downward_cell)
            
            row_eval, _ =  theano.scan(fn=ev,
                        outputs_info=[None, None, None, None, row_init_h, row_init_c], #Only have h_row and c_row as recurrent
                        sequences=[hs_prevlayer, cs_prevlayer, hs_prevrow, cs_prevrow], # run along the layer below and "upwards" row
                        non_sequences=None,
                        go_backwards=False) # We want to go forward, left to right, for now

            hs_nextlayer = row_eval[0]
            cs_nextlayer = row_eval[1]
            hs_nextrow = row_eval[2]
            cs_nextrow = row_eval[3]

            return hs_nextlayer, cs_nextlayer, hs_nextrow, cs_nextrow

        
        def ev_layer(htensor3_below, ctensor3_below):
            layer_eval, _ = theano.scan(fn=ev_row,
                            outputs_info=[None, None, h0_row, c0_row], #Only have row hs and cs as recurrent
                            sequences=[htensor3_below, ctensor3_below], #As sequences, we want the h and c rows below
                            non_sequences=None,
                            go_backwards=False) #We want to go forwards, up to down, for now

            hs_nextlayer = layer_eval[0]
            cs_nextlayer = layer_eval[1]

            return hs_nextlayer, cs_nextlayer
            

        def ev_block(htensor3_input):
            ctensor3_input = T.zeros_like(htensor3_input)

            self.n_layers = 5
            final_layer, _ = theano.scan(fn=ev_layer,
                                         outputs_info=[htensor3_input, ctensor3_input],
                                         n_steps=self.n_layers)

            return final_layer[0]

        return ev_block(input_tensor)
    
class lstm_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons, direction):
        self.input_neurons=input_neurons
        self.output_neurons=output_neurons
        self.direction=direction

        self.name = name
        self.neuron=single_lstm(name, input_neurons, output_neurons)

    
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

        # Discard the cell values:
        return lstm_preds[0]

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
        self.d_forward = lstm_layer(name + 'down_forward', input_neurons, output_neurons, True)
        self.d_backward = lstm_layer(name + 'down_backward', input_neurons, output_neurons, False)
        self.u_forward = lstm_layer(name + 'up_forward', input_neurons, output_neurons, True)
        self.u_backward = lstm_layer(name + 'up_backward', input_neurons, output_neurons, False)

        self.lstms = [self.d_forward, self.d_backward, self.u_forward, self.u_backward]


    def update_weights(self, update_list):
        prev = 0
        for lstm in self.lstms:
            cur = prev + lstm.weight_count()
            lstm.update_weights(update_list[prev:cur])
            prev = cur
            
    def weight_count(self):
        return sum([lstm.weight_count() for lstm in self.lstms])
        
    def get_theano_weights(self):
        return tuple(w for lstm in self.lstms for w in lstm.get_theano_weights())
    
    def get_python_weights(self):
        return tuple(w for lstm in self.lstms for w in lstm.get_python_weights())

    def __conc_wrapper(self, V, hs, layer_function):
        inp = T.concatenate((V, hs), axis=1)
        return layer_function(inp)
    
    def function(self, VM):
        init_hs = T.zeros((VM.shape[1], self.output_neurons))

        lstm_out_1, _ = theano.scan(fn=lambda a,b: self.__conc_wrapper(a,b,self.d_forward.function),
                                      outputs_info=init_hs,
                                      sequences=transpose_vm,
                                      non_sequences=None)
        lstm_out_2, _ = theano.scan(fn=lambda a,b: self.__conc_wrapper(a,b,self.d_backward.function),
                                      outputs_info=init_hs,
                                      sequences=transpose_vm,
                                      non_sequences=None)

        lstm_out_3, _ = theano.scan(fn=lambda a,b: self.__conc_wrapper(a,b,self.u_forward.function),
                                      outputs_info=init_hs,
                                      sequences=transpose_vm,
                                      non_sequences=None)

        lstm_out_4, _ = theano.scan(fn=lambda a,b: self.__conc_wrapper(a,b,self.u_backward.function),
                                      outputs_info=init_hs,
                                      sequences=transpose_vm,
                                      non_sequences=None)

        return T.concatenate((lstm_out_1, lstm_out_2, lstm_out_3, lstm_out_4), axis=2)

    
class linear_layer():

    training=False
    
    def __init__(self, name, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.name = name

        high_init = np.sqrt(6)/np.sqrt(input_neurons + output_neurons)
        low_init = -np.sqrt(6)/np.sqrt(input_neurons + output_neurons)
        
        if output_neurons == 1:
            self.weight_matrix_theano = T.dvector(name + '_weight')
            self.weight_matrix = np.random.uniform(low=low_init, high=high_init, size=self.input_neurons+1)
        else:
            self.weight_matrix_theano = T.dmatrix(name + '_weight')
            self.weight_matrix = np.random.uniform(low=low_init, high=high_init, size=(self.output_neurons, self.input_neurons+1))

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
        input_with_bias = T.concatenate((input_vector, [1]))
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


from theano import tensor as T
import theano
import numpy as np

network_ops = imp.load_source('network_ops', 'code/parsing/algorithms/network_ops.py')

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

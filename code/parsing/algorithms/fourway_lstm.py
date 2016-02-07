import numpy as np
from theano import tensor as T
import theano

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
        self.W_final = np.random.rand(self.hidden_dimension*4+1)

        self.W_forward_forget = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_forward_input = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_forward_cell = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_forward_output = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)

        self.W_backward_forget = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_backward_input = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_backward_cell = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_backward_output = np.random.rand(self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)


    '''
    Theano functions:
    '''

    # Prediction method for a single LSTM block:
    def __theano_lstm(self, x, h_prev, c_prev, W_forget, W_input, W_cell, W_output):

        input_vector = T.concatenate((x, h_prev))

        forget_gate = T.nnet.sigmoid(T.dot(W_forget, input_vector))
        input_gate = T.nnet.sigmoid(T.dot(W_input, input_vector))
        candidate_vector = T.tanh(T.dot(W_cell, input_vector))
        cell_state = forget_gate*c_prev + input_gate * candidate_vector

        output = T.nnet.sigmoid(T.dot(W_output, input_vector))
        h = output * T.tanh(cell_state)
        return h, cell_state


    # Prediction method for a layer of LSTM blocks:
    def __theano_lstm_layer(self, Vs, W_forget, W_input, W_cell, W_output, forwards=True):
        h0 = np.zeros(self.hidden_dimension)
        c0 = np.zeros(self.hidden_dimension)

        lstm_preds, _ = theano.scan(fn=self.__theano_lstm,
                            outputs_info=[h0,c0],
                            sequences=Vs,
                            non_sequences=[W_forget,
                                           W_input,
                                           W_cell,
                                           W_output],
                            go_backwards=not forwards)

        # Discard the cell values:
        return lstm_preds[0]

    def __pairwise_features(self, V, Vs, sentence_length):
        thingy, _ = theano.scan(fn=lambda x, y: T.concatenate([y,x,[1]]),
                                sequences=Vs,
                                non_sequences=V)
        
        #Make root feature and bias neuron:
        root_features = T.concatenate((V,T.ones(self.input_dimension + 1)))

        flat_version = thingy.flatten()
        with_root = T.concatenate((root_features, flat_version))
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.input_dimension*2+1))
        return in_shape

    def __theano_out_node(self, H, W_o):
        output_with_bias = T.concatenate((H, [1]))
        return T.dot(W_o, output_with_bias)
    
    def __theano_sentence_prediction(self, Vs, sentence_length, W_final,
                                     W_forget, W_input, W_cell, W_output):#, W_forget_b, W_input_b, W_cell_b, W_output_b):

        #Make pairwise features:
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, sentence_length])

        
        lstm_sidewards_forwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=pairwise_vs,
                                                 non_sequences=[W_forget, W_input, W_cell, W_output, 1])

        lstm_sidewards_backwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=pairwise_vs,
                                                 non_sequences=[W_forget, W_input, W_cell, W_output, 0])

        transpose_vs = pairwise_vs.transpose(1,0,2)

        lstm_downwards_forwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=transpose_vs,
                                                 non_sequences=[W_forget, W_input, W_cell, W_output, 1])

        lstm_downwards_backwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=transpose_vs,
                                                 non_sequences=[W_forget, W_input, W_cell, W_output, 0])

        full_lstm = T.concatenate((lstm_sidewards_forwards, lstm_sidewards_forwards, lstm_downwards_forwards.transpose(1,0,2), lstm_downwards_backwards.transpose(1,0,2)), axis=2)

        flatter_lstm = T.reshape(full_lstm, newshape=(sentence_length*(sentence_length+1), self.hidden_dimension*4))

        outputs, _ = theano.scan(fn=self.__theano_out_node,
                                 sequences=flatter_lstm,
                                 non_sequences=W_final)
        matrix_outputs = T.reshape(outputs, newshape=(sentence_length,sentence_length+1))

        
        return T.nnet.softmax(matrix_outputs)
        
    #For testing:
    def single_predict(self, sentence):
        Vs = T.dmatrix('Vs')
        W_forget = T.dmatrix('W_forget')
        W_input = T.dmatrix('W_input')
        W_cell = T.dmatrix('W_cell')
        W_output = T.dmatrix('W_output')
        W_final = T.vector('W_final')
        
        result = self.__theano_sentence_prediction(Vs, len(sentence), W_final, W_forget, W_input, W_cell, W_output)
        cgraph = theano.function(inputs=[Vs, W_final, W_forget, W_input, W_cell, W_output], on_unused_input='warn', outputs=result)

        print(np.array(sentence).shape)
        res = cgraph(sentence, self.W_final, self.W_forward_forget, self.W_forward_input, self.W_forward_cell, self.W_forward_output)
        print(res.shape)

        return res
    
def fit(features, labels, model_path=None, save_every_iteration=False):

    model = RNN()
    print(model.single_predict(features[0]))

def predict(sentence_list, model_path=None):
    predictions = []
    for sentence in sentence_list:
        predictions.append([])
        for token_feature in sentence:
            predictions[-1].append(np.random.uniform(0.0, 1.0, len(sentence)+1))

    return predictions

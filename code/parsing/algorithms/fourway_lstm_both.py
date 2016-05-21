import numpy as np
from theano import tensor as T
import theano
import pickle
import imp
from theano.tensor.shared_randomstreams import RandomStreams

superclass = imp.load_source('abstract_rnn', 'code/parsing/algorithms/abstract_rnn.py')
network_ops = imp.load_source('network_ops', 'code/parsing/algorithms/network_ops.py')
optimizers = imp.load_source('optimizers', 'code/parsing/algorithms/optimizers.py')

class FourwayLstm(superclass.RNN):

    '''
    Fields:
    '''
    char_hidden_dimension = 64
    char_input_dimension = 256

    sentence_input_dimension = 81
    hidden_dimension = 64
    
    '''
    Initialization:
    '''

    def __init__(self, optimizer_config_path):
        n_layers = 4

        self.input_lstm_forward_layer = network_ops.multilayer_lstm('input_layer_1', self.char_input_dimension,  self.sentence_input_dimension, self.char_hidden_dimension,self.hidden_dimension, True)
        self.input_lstm_backward_layer = network_ops.multilayer_lstm('input_layer_2', self.char_input_dimension, self.sentence_input_dimension, self.char_hidden_dimension, self.hidden_dimension, False)

        self.transition_layer = network_ops.fourdirectional_lstm_layer('input_layer_', self.hidden_dimension * 4 + 1, self.hidden_dimension)
        self.lstm_layers = [network_ops.fourdirectional_lstm_layer('layer_'+str(l),
                                                              self.hidden_dimension * 4,
                                                              self.hidden_dimension) for l in range(n_layers-1)]
        
        self.output_convolution = network_ops.linear_tensor_convolution_layer('output_layer', self.hidden_dimension*4, 1)
        
        self.layers = [self.input_lstm_forward_layer, self.input_lstm_backward_layer, self.transition_layer] + self.lstm_layers + [self.output_convolution]

        super().__init__('both', optimizer_config_path)

        
    '''
    Theano functions:
    '''

    def __pairwise_features(self, V, Vs, sentence_length):
        thingy, _ = theano.scan(fn=lambda x, y: T.concatenate([y,[0],x]),
                                sequences=Vs,
                                non_sequences=V)

        root_feature = T.concatenate((T.ones(1), T.zeros(self.hidden_dimension*2)))        
        root_features = T.concatenate((V,root_feature))

        flat_version = thingy.flatten()
        with_root = T.concatenate((root_features, flat_version))
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.hidden_dimension*4+1))
        return in_shape

    def theano_sentence_loss(self, Sentence, Chars, WordLengths, Gold):
        preds = self.theano_sentence_prediction(Sentence, Chars, WordLengths)
        losses = T.nnet.categorical_crossentropy(preds, Gold)
        return T.sum(losses)

    def theano_sentence_prediction(self, Sentence, Chars, WordLengths):

        input_lstm_res_f = self.input_lstm_forward_layer.function(Sentence, Chars, WordLengths)
        input_lstm_res_b = self.input_lstm_backward_layer.function(Sentence, Chars, WordLengths)
        input_combined = T.concatenate((input_lstm_res_f, input_lstm_res_b), axis=1)

        #Make pairwise features. This is really just "tensor product with concatenation instead of multiplication". Is there a command for that?
        full_matrix, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=input_combined,
                                  non_sequences=[input_combined, Sentence.shape[0]])

        if len(self.lstm_layers) > 0 and self.lstm_layers[0].training:
            srng = RandomStreams(seed=12345)
            full_matrix = T.switch(srng.binomial(size=(Sentence.shape[0], Sentence.shape[0]+1, self.hidden_dimension*4), p=0.5), full_matrix, 0)
        else:
            full_matrix = 0.5 * full_matrix

        full_matrix = self.transition_layer.function(full_matrix)
            
        for layer in self.lstm_layers:
            if layer.training:
                print("hah-train")
                full_matrix = T.switch(srng.binomial(size=(Sentence.shape[0], Sentence.shape[0]+1, self.hidden_dimension*4), p=0.5), full_matrix, 0)
            else:
                print("heh-notrain")
                full_matrix = 0.5 * full_matrix
            
            
            full_matrix = layer.function(full_matrix)
        
        final_matrix = self.output_convolution.function(full_matrix)

        return T.nnet.softmax(final_matrix)


def fit(features, labels, dev_features, dev_labels, model_path=None):
    optimizer_config_path = 'fourway_optimizer.config'    
    model = FourwayLstm(optimizer_config_path) #, list(features.keys()))
    #model.load(model_path)

    model.save_path = model_path
    model.train(features, labels, dev_features, dev_labels)
    
def predict(features, model_path=None):
    model = FourwayLstm(None)
    #model.load(model_path)

    predictions = model.predict(features)
    
    return predictions

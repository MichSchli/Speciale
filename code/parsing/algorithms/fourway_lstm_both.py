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

    sentence_input_dimension = 64
    
    hidden_dimension = 64
    
    '''
    Initialization:
    '''

    def __init__(self, optimizer_config_path):
        #n_layers = 2

        self.input_lstm_layer = network_ops.multilayer_lstm('input_layer_', self.char_input_dimension, self.char_hidden_dimension, self.sentence_input_dimension, self.hidden_dimension, True)

        #self.lstm_layers = [network_ops.fourdirectional_lstm_layer('layer_'+str(l),
        #                                                      self.hidden_dimension * 4,
        #                                                      self.hidden_dimension) for l in range(n_layers-1)]
        
        self.output_convolution = network_ops.linear_tensor_convolution_layer('output_layer', self.hidden_dimension*2, 1)
        
        self.layers = [self.input_lstm_layer] + [self.output_convolution]

        super().__init__('both', optimizer_config_path)

        
    '''
    Theano functions:
    '''
        
    def __pairwise_features(self, V, Vs, sentence_length):
        thingy, _ = theano.scan(fn=lambda x, y: T.concatenate([y,x]),
                                sequences=Vs,
                                non_sequences=V)
        
        #Make root feature:
        root_features = T.concatenate((V,T.ones(self.hidden_dimension)))

        flat_version = thingy.flatten()
        with_root = T.concatenate((root_features, flat_version))
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.hidden_dimension*2))
        return in_shape
   
    def theano_sentence_loss(self, Sentence, Chars, WordLengths, Gold):
        preds = self.theano_sentence_prediction(Sentence, Chars, WordLengths)
        losses = T.nnet.categorical_crossentropy(preds, Gold)
        return T.sum(losses)

    def theano_sentence_prediction(self, Sentence, Chars, WordLengths):

        input_lstm_res = self.input_lstm_layer.function(Sentence, Chars, WordLengths)

        #Make pairwise features. This is really just "tensor product with concatenation instead of multiplication". Is there a command for that?
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=input_lstm_res,
                                  non_sequences=[input_lstm_res, Sentence.shape[0]])
        
        final_matrix = self.output_convolution.function(pairwise_vs)

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

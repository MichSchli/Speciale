import numpy as np
from theano import tensor as T
import theano
import pickle
import imp

superclass = imp.load_source('abstract_rnn', 'code/parsing/algorithms/abstract_rnn.py')
network_ops = imp.load_source('network_ops', 'code/parsing/algorithms/network_ops.py')

class FourwayLstm(superclass.RNN):

    '''
    Fields:
    '''

    hidden_dimension = 16
    input_dimension = 64
    
    '''
    Initialization:
    '''

    def __init__(self):
        super().__init__()
        
        self.first_lstm_layer = network_ops.fourdirectional_lstm_layer('first_layer', self.input_dimension * 2, self.hidden_dimension)
        self.second_lstm_layer = network_ops.fourdirectional_lstm_layer('second_layer', self.hidden_dimension * 4, self.hidden_dimension)
        self.output_convolution = network_ops.linear_tensor_convolution_layer('output_layer', self.hidden_dimension * 4, 1)
        
        self.layers = [self.first_lstm_layer,  self.second_lstm_layer, self.output_convolution]

        
    '''
    Theano functions:
    '''
        
    def __pairwise_features(self, V, Vs, sentence_length):
        thingy, _ = theano.scan(fn=lambda x, y: T.concatenate([y,x]),
                                sequences=Vs,
                                non_sequences=V)
        
        #Make root feature:
        root_features = T.concatenate((V,T.ones(self.input_dimension)))

        flat_version = thingy.flatten()
        with_root = T.concatenate((root_features, flat_version))
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.input_dimension*2))
        return in_shape

    
   
    def theano_sentence_loss(self, Vs, sentence_length, gold):
        preds = self.theano_sentence_prediction(Vs, sentence_length)
        losses = T.nnet.categorical_crossentropy(preds, gold)
        return T.sum(losses)

    
    def theano_sentence_prediction(self, Vs, sentence_length):

        #Make pairwise features. This is really just "tensor product with concatenation instead of multiplication". Is there a command for that?
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, sentence_length])
        
        full_lstm = self.first_lstm_layer.function(pairwise_vs)
        full_lstm = self.second_lstm_layer.function(full_lstm)
        final_matrix = self.output_convolution.function(full_lstm)

        return T.nnet.softmax(final_matrix)

    
def fit(features, labels, model_path=None):
    model = FourwayLstm()
    model.load(model_path)

    model.save_path = model_path
    model.train(features, labels)
    
def predict(features, model_path=None):
    model = FourwayLstm()
    #model.load(model_path)

    predictions = model.batch_predict(features)
    
    return predictions

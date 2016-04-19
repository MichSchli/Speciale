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

    hidden_dimension = 64
    input_dimension = 64
    
    '''
    Initialization:
    '''

    def __init__(self, optimizer_config_path):
        n_layers = 3

        # Idea: Add a nonlinear tanh layer to the input.
        # Intuition: "Interpret this vector before comparing it to others."

        #For now, linear layer projecting to larger space to match network:
        self.input_lstm_layer = network_ops.fourdirectional_lstm_layer('input_layer_', self.input_dimension * 2, self.hidden_dimension)

        self.lstm_layers = [network_ops.fourdirectional_lstm_layer('layer_'+str(l),
                                                              self.hidden_dimension * 4,
                                                              self.hidden_dimension) for l in range(n_layers-1)]
        
        #self.first_lstm_layer = network_ops.fourdirectional_lstm_layer('first_layer', self.hidden_dimension * 4, self.hidden_dimension)
        #self.second_lstm_layer = network_ops.fourdirectional_lstm_layer('second_layer', self.hidden_dimension * 4, self.hidden_dimension)
        self.output_convolution = network_ops.linear_tensor_convolution_layer('output_layer', self.hidden_dimension * 4, 1)
        
        self.layers = [self.input_lstm_layer] + self.lstm_layers + [self.output_convolution]

        super().__init__('sentence', optimizer_config_path)

        
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

    
   
    def theano_sentence_loss(self, Vs, gold):
        preds = self.theano_sentence_prediction(Vs)
        losses = T.nnet.categorical_crossentropy(preds, gold)
        return T.sum(losses)

    
    def theano_sentence_prediction(self, Vs):

        #Make pairwise features. This is really just "tensor product with concatenation instead of multiplication". Is there a command for that?
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, Vs.shape[0]])

        
        if self.input_lstm_layer.training:
            srng = RandomStreams(seed=12345)
            #pairwise_vs = T.switch(srng.binomial(size=(sentence_length, sentence_length+1, self.input_dimension*2), p=0.5), pairwise_vs, 0)
        #else:
        #    pairwise_vs = 0.5*pairwise_vs
        #
        
        full_matrix = self.input_lstm_layer.function(pairwise_vs)

        for layer in self.lstm_layers:
            
            if self.input_lstm_layer.training:
                print("hah-train")
                full_matrix = T.switch(srng.binomial(size=(Vs.shape[0], Vs.shape[0]+1, self.hidden_dimension*4), p=0.5), full_matrix, 0)
            else:
                print("heh-notrain")
                full_matrix = 0.5 * full_matrix
            
            
            full_matrix = layer.function(full_matrix)

        '''
        if self.input_lstm_layer.training:
            full_matrix = T.switch(srng.binomial(size=(sentence_length, sentence_length+1, self.hidden_dimension*4), p=0.5), full_matrix, 0)
        else:
            full_matrix = 0.5 * full_matrix
        '''
        
        final_matrix = self.output_convolution.function(full_matrix)

        return T.nnet.softmax(final_matrix)

    

def fit(features, labels, dev_features, dev_labels, model_path=None):
    optimizer_config_path = 'fourway_optimizer.config'    
    model = FourwayLstm(optimizer_config_path) #, list(features.keys()))
    model.load(model_path)

    model.save_path = model_path
    model.train(features, labels, dev_features, dev_labels)
        
def predict(features, model_path=None):
    model = FourwayLstm(None)
    model.load(model_path)

    predictions = model.predict(features)
    
    return predictions

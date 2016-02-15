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

    hidden_dimension = 20
    input_dimension = 50

    learning_rate = 0.0005
    momentum = 0.1
    batch_size = 50

    error_margin = 0.000001
    
    '''
    Class methods:
    '''

    def __init__(self):
        super().__init__()
        
        self.first_lstm_layer = network_ops.fourdirectional_lstm_layer('first_layer', self.input_dimension * 2, self.hidden_dimension)
        self.second_lstm_layer = network_ops.fourdirectional_lstm_layer('second_layer', self.hidden_dimension * 4, self.hidden_dimension)
        self.output_convolution = network_ops.linear_tensor_convolution_layer('output_layer', self.hidden_dimension * 4, 1)

        self.layers = [self.first_lstm_layer, self.second_lstm_layer, self.output_convolution]

        print(self.get_weight_list())
        print(self.get_theano_weight_list())
        self.update_weights(self.get_weight_list())
        exit()
        
    '''
    Theano functions:
    '''
        
    def __pairwise_features(self, V, Vs, sentence_length):
        thingy, _ = theano.scan(fn=lambda x, y: T.concatenate([y,x]),
                                sequences=Vs,
                                non_sequences=V)
        
        #Make root feature and bias neuron:
        root_features = T.concatenate((V,T.ones(self.input_dimension)))

        flat_version = thingy.flatten()
        with_root = T.concatenate((root_features, flat_version))
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.input_dimension*2))
        return in_shape

    
    def __theano_predict_with_pad(self, Vs, sentence_length):

        preds = self.__theano_sentence_prediction(Vs, sentence_length)

        pad1 = T.zeros((sentence_length, self.max_words - sentence_length))
        pad2 = T.zeros((self.max_words - sentence_length, self.max_words + 1))

        padded_result = T.concatenate((preds, pad1), axis=1)
        padded_result = T.concatenate((padded_result, pad2), axis=0)        

        return padded_result

        
    def __theano_sentence_loss(self, Vs, sentence_length, gold):
        preds = self.__theano_sentence_prediction(Vs, sentence_length)

        gold = gold[:sentence_length, :sentence_length+1]
        losses = T.nnet.categorical_crossentropy(preds, gold)

        return T.sum(losses)


    def __theano_batch_loss(self, Vs, sentence_lengths, Gs):
        losses, __ = theano.scan(fn=self.__theano_sentence_loss,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths, Gs],
                                 non_sequences=None)

        return T.sum(losses)

    
    def __theano_batch_prediction(self, Vs, sentence_lengths):
        preds, __ = theano.scan(fn=self.__theano_predict_with_pad,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths],
                                non_sequences=None)

        return preds

    
    def __theano_sentence_prediction(self, Vs, sentence_length):

        Vs = Vs[:sentence_length]

        #Make pairwise features. This is really just "tensor product with concatenation instead of multiplication". Is there a command for that?
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, sentence_length])
        
        full_lstm = self.first_lstm_layer.function(pairwise_vs)

        full_lstm = self.second_lstm_layer.function(full_lstm)

        #Todo: scan
        #full_lstm1 = layer.function(full_lstm, W_forget_e[0], W_input_e[0], W_cell_e[0], W_output_e[0])
        #full_lstm2 = layer.function(full_lstm1, W_forget_e[1], W_input_e[1], W_cell_e[1], W_output_e[1])
        #full_lstm3 = layer.function(full_lstm2, W_forget_e[2], W_input_e[2], W_cell_e[2], W_output_e[2])
        
        final_matrix = self.output_convolution.function(full_lstm)

        return T.nnet.softmax(final_matrix)


    def theano_sgd(self, Vs, Ls, Gs):

        loss = self.__theano_batch_loss(Vs, Ls, Gs)

        weight_list = list(self.get_theano_weight_list())
        grads = T.grad(loss, weight_list)

        return grads



            
    '''
    Loss (move to abstract):
    '''
        
    def build_loss_graph(self):
        print("Building loss graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        Gs = T.tensor3('Gs')
        
        weight_list = self.get_theano_weight_list()
         
        result = self.__theano_batch_loss(Vs, Ls, Gs)

        input_list = [Vs, Ls, Gs] + list(weight_list)
        return theano.function(inputs=input_list, on_unused_input='warn', outputs=result)
        
    def batch_loss(self, sentences, lengths, golds):

        if self.loss_graph is None:
            self.loss_graph = self.build_loss_graph()

        weights = self.get_weight_list()        
        res = self.loss_graph(sentences, lengths, golds, *weights)

        return res

    '''
    Prediction (move to abstract):
    '''

    def build_predict_graph(self):
        print("Building prediction graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        weight_list = self.get_theano_weight_list()
        
        result = self.__theano_batch_prediction(Vs, Ls)
        input_list = [Vs, Ls] + list(weight_list)

        print("Done building graph.")
        return theano.function(inputs=input_list, on_unused_input='warn', outputs=result)
    
    
    def batch_predict(self, sentences):
        lengths = np.array([len(s) for s in sentences])
        lengths = lengths.astype(np.int32)

        sentences = self.pad_sentences(sentences)
        
        if self.predict_graph is None:
            self.predict_graph = self.build_predict_graph()

        weights = self.get_weight_list()
        res = self.predict_graph(sentences, lengths, *weights)

        out_sentences = []
        for sentence, length in zip(res, lengths):
            out_sentences.append(sentence[:length, :length+1])

        return out_sentences


    
def fit(features, labels, model_path=None):
    model = FourwayLstm()
    #model.load(model_path)

    model.save_path = model_path
    model.train(features, labels)
    
def predict(features, model_path=None):
    model = FourwayLstm()
    model.load(model_path)

    predictions = model.batch_predict(features)
    
    return predictions

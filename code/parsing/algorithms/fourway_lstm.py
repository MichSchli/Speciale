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

    learning_rate = 0.005
    momentum = 0.1
    batch_size = 50

    error_margin = 0.000001
    
    '''
    Class methods:
    '''

    def __init__(self):
        super().__init__()
        
        self.W_final = np.random.rand(self.hidden_dimension*4+1)

        n_lstm_layers = 3
        
        self.W_forget = np.random.rand(4, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_input = np.random.rand(4, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_cell = np.random.rand(4, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_output = np.random.rand(4, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        
    
        
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
        in_shape = T.reshape(with_root, newshape=(sentence_length+1,self.input_dimension*2+1))
        return in_shape

    
    def __theano_predict_with_pad(self, Vs, sentence_length, W_final,
                                     W_forget, W_input, W_cell, W_output):
        preds = self.__theano_sentence_prediction(Vs, sentence_length, W_final, W_forget, W_input, W_cell, W_output)

        pad1 = T.zeros((sentence_length, self.max_words - sentence_length))
        pad2 = T.zeros((self.max_words - sentence_length, self.max_words + 1))

        padded_result = T.concatenate((preds, pad1), axis=1)
        padded_result = T.concatenate((padded_result, pad2), axis=0)        

        return padded_result

        
    def __theano_sentence_loss(self, Vs, sentence_length, gold, W_final,
                                     W_forget, W_input, W_cell, W_output):
        preds = self.__theano_sentence_prediction(Vs, sentence_length, W_final, W_forget, W_input, W_cell, W_output)

        gold = gold[:sentence_length, :sentence_length+1]
        losses = T.nnet.categorical_crossentropy(preds, gold)

        return T.sum(losses)


    def __theano_batch_loss(self, Vs, sentence_lengths, Gs, W_final, W_forget, W_input, W_cell, W_output):
        losses, __ = theano.scan(fn=self.__theano_sentence_loss,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths, Gs],
                                non_sequences=[W_final, W_forget, W_input, W_cell, W_output])

        return T.sum(losses)

    
    def __theano_batch_prediction(self, Vs, sentence_lengths, W_final, W_forget, W_input, W_cell, W_output):
        preds, __ = theano.scan(fn=self.__theano_predict_with_pad,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths],
                                non_sequences=[W_final, W_forget, W_input, W_cell, W_output])

        return preds
    
    def __theano_sentence_prediction(self, Vs, sentence_length, W_final,
                                     W_forget, W_input, W_cell, W_output):

        Vs = Vs[:sentence_length]
        layer = network_ops.fourdirectional_lstm_layer(self.hidden_dimension)

        #Make pairwise features. This is really just "tensor product with concatenation instead of multiplication". Is there a command for that?
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, sentence_length])
        
        full_lstm = layer.function(pairwise_vs, W_forget, W_input, W_cell, W_output)

        flatter_lstm = T.reshape(full_lstm, newshape=(sentence_length*(sentence_length+1), self.hidden_dimension*4))

        outputs, _ = theano.scan(fn=network_ops.linear_layer,
                                 sequences=flatter_lstm,
                                 non_sequences=W_final)
        
        matrix_outputs = T.reshape(outputs, newshape=(sentence_length,sentence_length+1))

        return T.nnet.softmax(matrix_outputs)


    def theano_sgd(self, Vs, Ls, Gs,
                     W_final, W_forget,
                     W_input, W_cell, W_output,
                     W_final_prevupd, W_forget_prevupd,
                     W_input_prevupd, W_cell_prevupd,
                     W_output_prevupd):

        loss = self.__theano_batch_loss(Vs, Ls, Gs, W_final, W_forget, W_input, W_cell, W_output)

        grads = T.grad(loss, [W_final, W_forget, W_input, W_cell, W_output])

        newUpdFin = grads[0]*self.learning_rate + W_final_prevupd*self.momentum
        newUpdFor = grads[1]*self.learning_rate + W_forget_prevupd*self.momentum
        newUpdInp = grads[2]*self.learning_rate + W_input_prevupd*self.momentum
        newUpdCel = grads[3]*self.learning_rate + W_cell_prevupd*self.momentum
        newUpdOut = grads[4]*self.learning_rate + W_output_prevupd*self.momentum

        newFin = W_final - newUpdFin
        newFor = W_forget - newUpdFor
        newInp = W_input - newUpdInp
        newCel = W_cell - newUpdCel
        newOut = W_output - newUpdOut

        return newFin, newFor, newInp, newCel, newOut, newUpdFin, newUpdFor, newUpdInp, newUpdCel, newUpdOut

    def get_weight_list(self):
        return self.W_final, self.W_forget, self.W_input, self.W_cell, self.W_output
    
    def get_theano_weight_list(self):
        W_final = T.dvector('W_final')
        W_forget = T.dtensor3('W_forget')
        W_input = T.dtensor3('W_input')
        W_cell = T.dtensor3('W_cell')
        W_output = T.dtensor3('W_output')

        return W_final, W_forget, W_input, W_cell, W_output

    def get_initial_weight_updates(self):
        w_forget_upd = np.zeros_like(self.W_forget)
        w_input_upd = np.zeros_like(self.W_input)
        w_cell_upd = np.zeros_like(self.W_cell)
        w_output_upd = np.zeros_like(self.W_output)
        w_final_upd = np.zeros_like(self.W_final)

        return w_final_upd, w_forget_upd, w_input_upd, w_cell_upd, w_output_upd

    def update_weights(self, update_list):
        self.W_final = update_list[0]
        self.W_forget = update_list[1]
        self.W_input = update_list[2]
        self.W_cell = update_list[3]
        self.W_output = update_list[4]    

    def build_loss_graph(self):
        print("Building loss graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        Gs = T.tensor3('Gs')
        
        weight_list = self.get_theano_weight_list()
         
        result = self.__theano_batch_loss(Vs, Ls, Gs, *weight_list)

        input_list = [Vs, Ls, Gs] + weight_list
        return theano.function(inputs=input_list, on_unused_input='warn', outputs=result)
        
    def batch_loss(self, sentences, lengths, golds):

        if self.loss_graph is None:
            self.loss_graph = self.build_loss_graph
        
        res = self.loss_graph(sentences, lengths, golds, self.W_final, self.W_forget, self.W_input, self.W_cell, self.W_output)

        return res

    def build_predict_graph(self):
        print("Building prediction graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        weight_list = self.get_theano_weight_list()
        
        result = self.__theano_batch_prediction(Vs, Ls, *weight_list)
        input_list = [Ls, Vs] + weight_list
        return theano.function(inputs=input_list, on_unused_input='warn', outputs=result)
    
    
    def batch_predict(self, sentences):
        lengths = np.array([len(s) for s in sentences])
        lengths = lengths.astype(np.int32)

        sentences = self.pad_sentences(sentences)

        if self.predict_graph is None:
            self.predict_graph = self.build_predict_graph()

        res = self.predict_graph(sentences, lengths, self.W_final, self.W_forget, self.W_input, self.W_cell, self.W_output)

        out_sentences = []
        for sentence, length in zip(res, lengths):
            out_sentences.append(sentence[:length, :length+1])

        return out_sentences


    
def fit(features, labels, model_path=None):

    model = FourwayLstm()

    model.load(model_path)

    model.save_path = model_path
    model.train(features, labels)
    
def predict(features, model_path=None):
    model = FourwayLstm()
    model.load(model_path)

    predictions = model.batch_predict(features)
    
    return predictions

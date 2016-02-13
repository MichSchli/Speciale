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

        
    def __theano_sentence_loss(self, Vs, sentence_length, gold, W_final,
                                     W_forget, W_input, W_cell, W_output,
                               W_forget_e, W_input_e, W_cell_e, W_output_e):
        preds = self.__theano_sentence_prediction(Vs, sentence_length, W_final, W_forget, W_input, W_cell, W_output,
                               W_forget_e, W_input_e, W_cell_e, W_output_e)

        gold = gold[:sentence_length, :sentence_length+1]
        losses = T.nnet.categorical_crossentropy(preds, gold)

        return T.sum(losses)


    def __theano_batch_loss(self, Vs, sentence_lengths, Gs, W_final, W_forget, W_input, W_cell, W_output,W_forget_e, W_input_e, W_cell_e, W_output_e):
        losses, __ = theano.scan(fn=self.__theano_sentence_loss,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths, Gs],
                                non_sequences=[W_final, W_forget, W_input, W_cell, W_output,W_forget_e, W_input_e, W_cell_e, W_output_e])

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
        
        #layer = network_ops.fourdirectional_lstm_layer(self.hidden_dimension)

        full_lstm = self.first_lstm_layer.function(pairwise_vs)

        #full_lstm = self.second_lstm_layer.function(full_lstm)

        #Todo: scan
        #full_lstm1 = layer.function(full_lstm, W_forget_e[0], W_input_e[0], W_cell_e[0], W_output_e[0])
        #full_lstm2 = layer.function(full_lstm1, W_forget_e[1], W_input_e[1], W_cell_e[1], W_output_e[1])
        #full_lstm3 = layer.function(full_lstm2, W_forget_e[2], W_input_e[2], W_cell_e[2], W_output_e[2])

        #conv = network_ops.linear_tensor_convolution_layer(1)

        final_matrix = self.output_convolution.function(full_lstm)

        return T.nnet.softmax(final_matrix)


    def theano_sgd(self, Vs, Ls, Gs,
                     W_final, W_forget, W_input, W_cell, W_output,
                     W_forget_e, W_input_e, W_cell_e, W_output_e,
                     W_final_prevupd, W_forget_prevupd, W_input_prevupd, W_cell_prevupd, W_output_prevupd,
                     W_forget_prevupd_e, W_input_prevupd_e, W_cell_prevupd_e, W_output_prevupd_e):

        loss = self.__theano_batch_loss(Vs, Ls, Gs, W_final, W_forget, W_input, W_cell, W_output,W_forget_e, W_input_e, W_cell_e, W_output_e)

        grads = T.grad(loss, [W_final, W_forget, W_input, W_cell, W_output,W_forget_e, W_input_e, W_cell_e, W_output_e])

        newUpdFin = grads[0]*self.learning_rate + W_final_prevupd*self.momentum
        newUpdFor = grads[1]*self.learning_rate + W_forget_prevupd*self.momentum
        newUpdInp = grads[2]*self.learning_rate + W_input_prevupd*self.momentum
        newUpdCel = grads[3]*self.learning_rate + W_cell_prevupd*self.momentum
        newUpdOut = grads[4]*self.learning_rate + W_output_prevupd*self.momentum

        newUpdFor_e = grads[5]*self.learning_rate + W_forget_prevupd_e*self.momentum
        newUpdInp_e = grads[6]*self.learning_rate + W_input_prevupd_e*self.momentum
        newUpdCel_e = grads[7]*self.learning_rate + W_cell_prevupd_e*self.momentum
        newUpdOut_e = grads[8]*self.learning_rate + W_output_prevupd_e*self.momentum
        
        newFin = W_final - newUpdFin
        newFor = W_forget - newUpdFor
        newInp = W_input - newUpdInp
        newCel = W_cell - newUpdCel
        newOut = W_output - newUpdOut

        newFor_e = W_forget_e - newUpdFor_e
        newInp_e = W_input_e - newUpdInp_e
        newCel_e = W_cell_e - newUpdCel_e
        newOut_e = W_output_e - newUpdOut_e

        return newFin, newFor, newInp, newCel, newOut, newFor_e, newInp_e, newCel_e, newOut_e, newUpdFin, newUpdFor, newUpdInp, newUpdCel, newUpdOut, newUpdFor_e, newUpdInp_e, newUpdCel_e, newUpdOut_e

    def get_weight_list(self):
        #return self.W_final, self.W_forget, self.W_input, self.W_cell, self.W_output, self.W_forget_e, self.W_input_e, self.W_cell_e, self.W_output_e
        return self.first_lstm_layer.get_python_weights() + self.second_lstm_layer.get_python_weights() + self.output_convolution.get_python_weights()
    
    def get_theano_weight_list(self):
        #W_final = T.dvector('W_final')
        #W_forget = T.dtensor3('W_forget')
        #W_input = T.dtensor3('W_input')
        #W_cell = T.dtensor3('W_cell')
        #W_output = T.dtensor3('W_output')

        #W_forget_e = T.dtensor4('W_forget_e')
        #W_input_e = T.dtensor4('W_input_e')
        #W_cell_e = T.dtensor4('W_cell_e')
        #W_output_e = T.dtensor4('W_output_e')

        return self.first_lstm_layer.get_theano_weights() + self.second_lstm_layer.get_theano_weights() + self.output_convolution.get_theano_weights()

    def get_initial_weight_updates(self):
        w_forget_upd = np.zeros_like(self.W_forget)
        w_input_upd = np.zeros_like(self.W_input)
        w_cell_upd = np.zeros_like(self.W_cell)
        w_output_upd = np.zeros_like(self.W_output)
        w_final_upd = np.zeros_like(self.W_final)

        w_forget_upd_e = np.zeros_like(self.W_forget_e)
        w_input_upd_e = np.zeros_like(self.W_input_e)
        w_cell_upd_e = np.zeros_like(self.W_cell_e)
        w_output_upd_e = np.zeros_like(self.W_output_e)
        
        return w_final_upd, w_forget_upd, w_input_upd, w_cell_upd, w_output_upd, w_forget_upd_e, w_input_upd_e, w_cell_upd_e, w_output_upd_e

    def update_weights(self, update_list):
        self.W_final = update_list[0]
        self.W_forget = update_list[1]
        self.W_input = update_list[2]
        self.W_cell = update_list[3]
        self.W_output = update_list[4]

        self.W_forget_e = update_list[5]
        self.W_input_e = update_list[6]
        self.W_cell_e = update_list[7]
        self.W_output_e = update_list[8]

    def build_loss_graph(self):
        print("Building loss graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        Gs = T.tensor3('Gs')
        
        weight_list = self.get_theano_weight_list()
         
        result = self.__theano_batch_loss(Vs, Ls, Gs, *weight_list)

        input_list = [Vs, Ls, Gs] + list(weight_list)
        return theano.function(inputs=input_list, on_unused_input='warn', outputs=result)
        
    def batch_loss(self, sentences, lengths, golds):

        if self.loss_graph is None:
            self.loss_graph = self.build_loss_graph()
        
        res = self.loss_graph(sentences, lengths, golds, self.W_final, self.W_forget, self.W_input, self.W_cell, self.W_output,
                              self.W_forget_e, self.W_input_e, self.W_cell_e, self.W_output_e)

        return res

    def build_predict_graph(self):
        print("Building prediction graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        weight_list = self.get_theano_weight_list()

        print(weight_list)
        
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
    #model.load(model_path)

    predictions = model.batch_predict(features)
    
    return predictions

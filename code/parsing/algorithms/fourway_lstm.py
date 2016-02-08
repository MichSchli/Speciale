import numpy as np
from theano import tensor as T
import theano

class RNN():

    '''
    Fields:
    '''

    hidden_dimension = 2
    input_dimension = 50

    learning_rate = 0.01
    momentum = 0.1
    
    '''
    Class methods:
    '''

    def __init__(self):
        self.W_final = np.random.rand(self.hidden_dimension*4+1)

        n_lstm_layers = 4
        
        self.W_forget = np.random.rand(n_lstm_layers, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_input = np.random.rand(n_lstm_layers, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_cell = np.random.rand(n_lstm_layers, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)
        self.W_output = np.random.rand(n_lstm_layers, self.hidden_dimension, self.input_dimension*2 + self.hidden_dimension + 1)


    def __pad_sentences(self, sentence_list):
        longest_sentence = max([len(x) for x  in sentence_list])

        self.max_words = longest_sentence
        
        new_sentences = np.zeros((len(sentence_list), longest_sentence, len(sentence_list[0][0])))

        for i, sentence in enumerate(sentence_list):
            new_sentences[i, :len(sentence), :] = sentence

        return new_sentences

    def __pad_golds(self, sentence_labels):
        longest_sentence = max([len(x) for x  in sentence_labels])

        new_labels = np.zeros((len(sentence_labels), longest_sentence, longest_sentence+1))

        for i, sentence in enumerate(sentence_labels):
            for j,label in enumerate(sentence):
                new_labels[i,j,:label.shape[0]] = label

        return np.array(new_labels)

    
        
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


    def __theano_batch_loss(self, Vs, sentence_lengths,W_final, W_forget, W_input, W_cell, W_output, Gs):
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

        #Make pairwise features:
        pairwise_vs, _ = theano.scan(fn=self.__pairwise_features,
                                  outputs_info=None,
                                  sequences=Vs,
                                  non_sequences=[Vs, sentence_length])

        
        lstm_sidewards_forwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=pairwise_vs,
                                                 non_sequences=[W_forget[0], W_input[0], W_cell[0], W_output[0], 1])

        lstm_sidewards_backwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=pairwise_vs,
                                                 non_sequences=[W_forget[1], W_input[1], W_cell[1], W_output[1], 0])

        transpose_vs = pairwise_vs.transpose(1,0,2)

        lstm_downwards_forwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=transpose_vs,
                                                 non_sequences=[W_forget[2], W_input[2], W_cell[2], W_output[2], 1])

        lstm_downwards_backwards, _ = theano.scan(fn=self.__theano_lstm_layer,
                                                 outputs_info=None,
                                                 sequences=transpose_vs,
                                                 non_sequences=[W_forget[3], W_input[3], W_cell[3], W_output[3], 0])

        full_lstm = T.concatenate((lstm_sidewards_forwards, lstm_sidewards_forwards, lstm_downwards_forwards.transpose(1,0,2), lstm_downwards_backwards.transpose(1,0,2)), axis=2)

        flatter_lstm = T.reshape(full_lstm, newshape=(sentence_length*(sentence_length+1), self.hidden_dimension*4))

        outputs, _ = theano.scan(fn=self.__theano_out_node,
                                 sequences=flatter_lstm,
                                 non_sequences=W_final)
        matrix_outputs = T.reshape(outputs, newshape=(sentence_length,sentence_length+1))

        return T.nnet.softmax(matrix_outputs)


    def __theano_sgd(self, Vs, Ls, Gs,
                     W_final, W_forget,
                     W_input, W_cell, W_output,
                     W_final_prevupd, W_forget_prevupd,
                     W_input_prevupd, W_cell_prevupd,
                     W_output_prevupd):

        loss = self.__theano_batch_loss(Vs, Ls, W_final, W_forget, W_input, W_cell, W_output, Gs)

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
        
                       
    
    #For testing:
    def single_predict(self, sentences, golds):

        #Pad the sentences to allow use of tensor rather than list in theano:
        lengths = [len(s) for s in sentences]
        sentences = self.__pad_sentences(sentences)
        golds = self.__pad_golds(golds)
        
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        W_forget = T.dtensor3('W_forget')
        W_input = T.dtensor3('W_input')
        W_cell = T.dtensor3('W_cell')
        W_output = T.dtensor3('W_output')
        W_final = T.dvector('W_final')
        Gs = T.tensor3('Gs')
        
        result = self.__theano_sgd(Vs, Ls, Gs, W_final, W_forget, W_input, W_cell, W_output, W_final, W_forget, W_input, W_cell, W_output)
        cgraph = theano.function(inputs=[Vs, Ls, Gs, W_final, W_forget, W_input, W_cell, W_output], on_unused_input='warn', outputs=result)

        print(np.array(sentences).shape)
        res = cgraph(sentences, lengths, golds, self.W_final, self.W_forget, self.W_input, self.W_cell, self.W_output)
        print(res.shape)

        return res
    
def fit(features, labels, model_path=None, save_every_iteration=False):

    model = RNN()
    print(model.single_predict(features[:3], labels[:3]))

def predict(sentence_list, model_path=None):
    predictions = []
    for sentence in sentence_list:
        predictions.append([])
        for token_feature in sentence:
            predictions[-1].append(np.random.uniform(0.0, 1.0, len(sentence)+1))

    return predictions

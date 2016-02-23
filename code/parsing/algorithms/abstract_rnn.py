import numpy as np
from theano import tensor as T
import theano
import pickle
import os.path
import imp
import sys

sys.setrecursionlimit(10000)

optimizers = imp.load_source('optimizers', 'code/parsing/algorithms/optimizers.py')
io = imp.load_source('io', 'code/common/io.py')

class RNN():

    predict_graph = None
    
    def __init__(self, optimizer_config_path):
        if optimizer_config_path is not None:
            self.optimizer = optimizers.from_config(io.read_config_file(optimizer_config_path))

            self.optimizer.set_initial_weights(self.get_weight_list())
            self.optimizer.set_loss_function(self.build_loss_graph())
            self.optimizer.set_gradient_function(self.build_single_gradient_graph())
            self.optimizer.set_update_function(self.update_function)

            self.optimizer.initialize()

    '''
    Update:
    '''
    def update_function(self, weights):
        self.update_weights(weights)
        self.save(self.save_path)

    '''
    Weight functions:
    '''
    
    def get_weight_list(self):
        return [weight for layer in self.layers for weight in layer.get_python_weights()]

    
    def get_theano_weight_list(self):
        return [weight for layer in self.layers for weight in layer.get_theano_weights()]

    
    def update_weights(self, update_list):
        prev_count = 0
        for layer in self.layers:
            current_count = prev_count + layer.weight_count()
            layer.update_weights(update_list[prev_count:current_count])
            prev_count = current_count

    
    '''
    Prediction
    '''

    def __theano_predict_with_pad(self, Vs, sentence_length):
        Vs = Vs[:sentence_length]
        preds = self.theano_sentence_prediction(Vs, sentence_length)

        pad1 = T.zeros((sentence_length, self.max_words - sentence_length))
        pad2 = T.zeros((self.max_words - sentence_length, self.max_words + 1))

        padded_result = T.concatenate((preds, pad1), axis=1)
        padded_result = T.concatenate((padded_result, pad2), axis=0)        

        return padded_result

    
    def theano_batch_prediction(self, Vs, sentence_lengths):
        preds, __ = theano.scan(fn=self.__theano_predict_with_pad,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths],
                                non_sequences=None)

        return preds

    
    def build_predict_graph(self, saved_graph=None):
        if saved_graph is not None:
            exists = os.path.isfile(saved_graph)
            if exists:
                print("Loading graph...")
                return self.load_graph(saved_graph)

        print("Building prediction graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        weight_list = self.get_theano_weight_list()
        
        result = self.theano_batch_prediction(Vs, Ls)
        input_list = [Vs, Ls] + list(weight_list)

        cgraph = theano.function(inputs=input_list, on_unused_input='warn', outputs=result, mode='FAST_RUN')

        print("Done building graph.")
        #cgraph.profile.print_summary()

        if saved_graph is not None:
            self.save_graph(cgraph, saved_graph)
        
        return cgraph
    
    def batch_predict(self, sentences):
        lengths = np.array([len(s) for s in sentences])
        lengths = lengths.astype(np.int32)

        sentences = self.pad_sentences(sentences)
        
        if self.predict_graph is None:
            self.predict_graph = self.build_predict_graph()

        print("Predicting...")
        weights = self.get_weight_list()
        res = self.predict_graph(sentences, lengths, *weights)

        out_sentences = []
        for sentence, length in zip(res, lengths):
            out_sentences.append(sentence[:length, :length+1])

        return out_sentences

            
    '''
    Loss:
    '''

    
    def __theano_loss_with_pad(self, Vs, sentence_length, gold):
        Vs  = Vs[:sentence_length]
        gold = gold[:sentence_length, :sentence_length+1]
        return self.theano_sentence_loss(Vs, sentence_length, gold)

    
    def theano_batch_loss(self, Vs, sentence_lengths, Gs):
        losses, __ = theano.scan(fn=self.__theano_loss_with_pad,
                                outputs_info=None,
                                sequences=[Vs,sentence_lengths, Gs],
                                non_sequences=None)

        return T.sum(losses)

    
    def build_loss_graph(self, saved_graph=None):
        if saved_graph is not None:
            exists = os.path.isfile(saved_graph)
            if exists:
                print("Loading loss graph...")
                return self.load_graph(saved_graph)
        
        print("Building loss graph...")
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        Gs = T.tensor3('Gs')
        
        weight_list = self.get_theano_weight_list()
         
        result = self.theano_batch_loss(Vs, Ls, Gs)

        input_list = [Vs, Ls, Gs] + list(weight_list)
        cgraph = theano.function(inputs=input_list, on_unused_input='warn', outputs=result) #, profile=True)

        print("Done building graph.")
        #cgraph.profile.print_summary()
        
        if saved_graph is not None:
            self.save_graph(cgraph, saved_graph)
        
        return cgraph
    
    def batch_loss(self, sentences, lengths, golds):

        if self.loss_graph is None:
            self.loss_graph = self.build_loss_graph()

        weights = self.get_weight_list()
        print("Predicting...")
        res = self.loss_graph(sentences, lengths, golds, *weights)

        return res

            
    '''
    SGD:
    '''

    def build_single_gradient_graph(self):
        print("Building gradient graph...")
        
        V = T.matrix('V')
        L = T.iscalar('L')
        G = T.dmatrix('G')

        theano_weight_list = self.get_theano_weight_list()

        loss = self.theano_sentence_loss(V, L, G)
        grads = T.grad(loss, theano_weight_list)

        input_list = [V, L, G] + list(theano_weight_list)
        cgraph = theano.function(inputs=input_list, outputs=grads, mode='FAST_RUN')
        
        print("Done building graph")

        return cgraph
        
    
    def build_sgd_graph(self, saved_graph=None):
        if saved_graph is not None:
            exists = os.path.isfile(saved_graph)
            if exists:
                print("Loading gradient graph...")
                return self.load_graph(saved_graph)
                    
        print("Building gradient graph...")
        
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        Gs = T.dtensor3('Gs')

        theano_weight_list = self.get_theano_weight_list()

        loss = self.theano_batch_loss(Vs, Ls, Gs)
        grads = T.grad(loss, theano_weight_list)

        input_list = [Vs, Ls, Gs] + list(theano_weight_list)
        cgraph = theano.function(inputs=input_list, outputs=grads, mode='FAST_RUN')
        
        print("Done building graph")

        if saved_graph is not None:
            self.save_graph(cgraph, saved_graph)
        
        return cgraph

    
    def train(self, sentences, labels):

        longest_sentence = max([len(x) for x  in sentences])
        self.max_words = longest_sentence

        self.optimizer.set_training_data(sentences, labels)
        self.optimizer.set_development_data(sentences, labels)

        self.optimizer.update()



    '''
    Persistence:
    '''

    def save_graph(self, graph, filename):
        outfile = open(filename, 'wb')
        pickle.dump(graph, outfile)
        outfile.close()

    def load_graph(self, filename):
        infile = open(filename, 'rb')
        graph = pickle.load(infile)
        infile.close()

        return graph
        
        
    def save(self, filename):
        store_list = self.get_weight_list()
        
        outfile1 = open(filename, 'wb')
        pickle.dump(store_list, outfile1)
        outfile1.close()

        
    def load(self, filename):
        infile = open(filename, 'rb')
        store_list = pickle.load(infile)
        infile.close()

        self.update_weights(store_list)

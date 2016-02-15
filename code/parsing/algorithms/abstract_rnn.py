import numpy as np
from theano import tensor as T
import theano
import pickle

class RNN():

    sgd_graph = None
    loss_graph = None
    predict_graph = None
    
    def __init__(self):
        pass
    
    '''
    Chunking and padding:
    '''

    def chunk(self, l):
        return np.array(list(zip(*[iter(l)]*self.batch_size)))

        
    def pad_sentences(self, sentence_list):
        longest_sentence = max([len(x) for x  in sentence_list])

        self.max_words = longest_sentence
        
        new_sentences = np.zeros((len(sentence_list), longest_sentence, len(sentence_list[0][0])))

        for i, sentence in enumerate(sentence_list):
            new_sentences[i, :len(sentence), :] = sentence

        return new_sentences

    def pad_golds(self, sentence_labels):
        longest_sentence = max([len(x) for x  in sentence_labels])

        new_labels = np.zeros((len(sentence_labels), longest_sentence, longest_sentence+1))

        for i, sentence in enumerate(sentence_labels):
            for j,label in enumerate(sentence):
                new_labels[i,j,:label.shape[0]] = label

        return np.array(new_labels)

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
    SGD:
    '''


    def build_sgd_graph(self):
        print("Building graph...")
        
        Vs = T.dtensor3('Vs')
        Ls = T.ivector('Ls')
        Gs = T.dtensor3('Gs')

        theano_weight_list = self.get_theano_weight_list()

        input_list = [Vs, Ls, Gs] + list(theano_weight_list)
        grads = self.theano_sgd(Vs, Ls, Gs)
        
        cgraph = theano.function(inputs=input_list, outputs=grads)
        
        print("Done building graph")
        
        return cgraph
    
    def train(self, sentences, labels):

        if self.sgd_graph is None:
            self.sgd_graph = self.build_sgd_graph()
                    
        lengths = np.array([len(s) for s in sentences])
        lengths = lengths.astype(np.int32)

        sentences = self.pad_sentences(sentences)
        labels = self.pad_golds(labels)
        
        length_chunks = self.chunk(lengths)
        sentence_chunks = self.chunk(sentences)
        label_chunks = self.chunk(labels)

        current_loss = self.batch_loss(sentences, lengths, labels)
        prev_loss = current_loss +1

        iteration_counter = 1

        weight_list = self.get_weight_list()
        updates = [np.zeros_like(weight) for weight in weight_list]
        
        while(prev_loss - current_loss > self.error_margin and iteration_counter < 100):
            prev_loss = current_loss
            print("Running gradient descent at iteration "+str(iteration_counter)+". Current loss: "+str(prev_loss))
            iteration_counter += 1
            
            for data_batch, length_batch, label_batch in zip(sentence_chunks, length_chunks, label_chunks):
                gradients = self.sgd_graph(data_batch, length_batch, label_batch, *weight_list)
                print('.')

                for i in range(len(weight_list)):
                    updates[i] = self.learning_rate * gradients[i] + self.momentum * updates[i]
                    weight_list[i] -= updates[i]

                self.update_weights(weight_list)
            
            current_loss = self.batch_loss(sentences, lengths, labels)
            self.save(self.save_path)


    '''
    Persistence:
    '''
    
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

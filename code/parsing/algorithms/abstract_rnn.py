import numpy as np
from theano import tensor as T
import theano

class RNN():

    sgd_graph = None
    
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
    SGD:
    '''


    def build_sgd_graph(self):
        print("Building graph...")
        
        Vs = T.dtensor4('Vs')
        Ls = T.imatrix('Ls')
        Gs = T.tensor4('Gs')

        theano_weight_list = self.get_theano_weight_list()
        initial_weight_updates = self.get_initial_weight_updates()
        function_output_init = theano_weight_list + initial_weight_updates

        results, _ =  theano.scan(fn=self.theano_sgd,
                                  outputs_info=function_output_init,
                                  sequences=[Vs, Ls, Gs],
                                  non_sequences=None)

        final_output_list = [results[j][-1] for j in range(len(theano_weight_list))]

        input_list = [Vs, Ls, Gs] + list(theano_weight_list)
        cgraph = theano.function(inputs=input_list, outputs=final_output_list)

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
        
        while(prev_loss - current_loss > self.error_margin and iteration_counter < 11):

            prev_loss = current_loss
            print("Running gradient descent at iteration "+str(iteration_counter)+". Current loss: "+str(prev_loss))
            iteration_counter += 1
            
            weight_list = self.get_weight_list()
            out_list = self.sgd_graph(sentence_chunks, length_chunks, label_chunks, *weight_list)

            self.update_weights(out_list)
            
            current_loss = self.batch_loss(sentences, lengths, labels)
            self.save(self.save_path)

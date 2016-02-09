class RNN():
    
    def __init__(self):
        pass

    '''
    Chunking and padding:
    '''

    def __chunk(self, l):
        return np.array(list(zip(*[iter(l)]*self.batch_size)))

        
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

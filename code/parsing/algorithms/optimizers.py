import sys
import numpy as np

class Optimizer():

    def __init__(self, algorithm):
        assert(algorithm.loss_graph is not None)
        assert(algorithm.sgd_graph is not None)
        
        self.algorithm = algorithm
        self.loss_function = algorithm.loss_graph
        self.gradient_function = algorithm.sgd_graph
        
        self.weights = algorithm.get_weight_list()

    def do_update(self):
        self.algorithm.update_weights(self.weights)
        self.algorithm.save(self.algorithm.save_path)

        
class MinibatchOptimizer(Optimizer):

    max_iterations = 100
    error_margin = 0.0000001

    def __init__(self, algorithm, batch_size):
        super().__init__(algorithm)

        self.batch_size = batch_size

    def chunk(self, l):
        return np.array(list(zip(*[iter(l)]*self.batch_size)))
        
    def update(self, sentences, lengths, labels):
        length_chunks = self.chunk(lengths)
        sentence_chunks = self.chunk(sentences)
        label_chunks = self.chunk(labels)

        current_loss = self.loss_function(sentences, lengths, labels, *self.weights)
        prev_loss = current_loss +1

        iteration_counter = 1

        self.updates = [np.zeros_like(weight) for weight in self.weights]
        
        while(prev_loss - current_loss > self.error_margin and iteration_counter < self.max_iterations):
            prev_loss = current_loss
            print("Running optimizer at iteration "+str(iteration_counter)+". Current loss: "+str(prev_loss))
            iteration_counter += 1
            
            for data_batch, length_batch, label_batch in zip(sentence_chunks, length_chunks, label_chunks):
                self.batch_update(data_batch, length_batch, label_batch)

                for i, update in enumerate(self.updates):
                    self.weights[i] += self.updates[i]

            print('')
            self.do_update()

            current_loss = self.loss_function(sentences, lengths, labels, *self.weights)


class StochasticGradientDescent(MinibatchOptimizer):

    def __init__(self, algorithm, batch_size, learning_rate, momentum, verbose=False):
        super().__init__(algorithm, batch_size)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.verbose = verbose

    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.gradient_function(data_batch, length_batch, label_batch, *self.weights)

        if self.verbose:
            print('.', end='', flush=True)

        for i, gradient in enumerate(gradients):
            self.updates[i] = -self.learning_rate*gradient + self.momentum * self.updates[i]
            


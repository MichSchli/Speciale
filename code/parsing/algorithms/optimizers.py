import sys
import numpy as np

class Optimizer():

    gradient_clipping_factor = 15

    use_gradient_noise = True
    gradient_noise_eta = 0.01
    gradient_noise_gamma = 0.55
    
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


    def process_gradients(self, gradients):
        if self.use_gradient_noise:
            std_dev = np.sqrt(self.gradient_noise_eta/(self.current_iteration**self.gradient_noise_gamma))

            gradients = [g + np.random.normal(0, std_dev, size=g.shape) for g in gradients]
        
        if self.gradient_clipping_factor is not None:
            gradient_l2_norm = np.sqrt(sum([np.square(g).sum() for g in gradients]))
            if gradient_l2_norm > self.gradient_clipping_factor:
                gradients = [g*float(self.gradient_clipping_factor)/gradient_l2_norm for g in gradients]

        return gradients
        
class MinibatchOptimizer(Optimizer):

    max_iterations = 10
    error_margin = 0.0000001
    normalize_batches = True
    gradient_clipping_factor = 15
    
    def __init__(self, algorithm, batch_size):
        super().__init__(algorithm)

        self.batch_size = batch_size

    def process_gradients(self, gradients):
        if self.normalize_batches:
            gradients = [g/self.batch_size for g in gradients]
        
        gradients = super().process_gradients(gradients)
        return gradients
        
    def chunk(self, l):
        return np.array([l[i:i+self.batch_size] for i in range(0, len(l), self.batch_size)])
        
    def update(self, sentences, lengths, labels):
        self.current_iteration = 1
        
        length_chunks = self.chunk(lengths)
        sentence_chunks = self.chunk(sentences)
        label_chunks = self.chunk(labels)

        current_loss = self.loss_function(sentences, lengths, labels, *self.weights)
        prev_loss = current_loss +1

        self.updates = [np.zeros_like(weight) for weight in self.weights]
        
        while(self.current_iteration < self.max_iterations):
            prev_loss = current_loss
            print("Running optimizer at iteration "+str(self.current_iteration)+". Current loss: "+str(prev_loss))
            self.current_iteration += 1
            
            for data_batch, length_batch, label_batch in zip(sentence_chunks, length_chunks, label_chunks):
                self.batch_update(data_batch, length_batch, label_batch)

                for i, update in enumerate(self.updates):
                    self.weights[i] += self.updates[i]

            print('')
            self.do_update()

            current_loss = self.loss_function(sentences, lengths, labels, *self.weights)
            #print(current_loss)


class StochasticGradientDescent(MinibatchOptimizer):

    def __init__(self, algorithm, batch_size, learning_rate, momentum, verbose=False):
        super().__init__(algorithm, batch_size)

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.verbose = verbose

    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.gradient_function(data_batch, length_batch, label_batch, *self.weights)
        gradients = self.process_gradients(gradients)
        
        for i, gradient in enumerate(gradients):
            self.updates[i] = -self.learning_rate*gradient + self.momentum * self.updates[i]

        if self.verbose:
            print('.', end='', flush=True)


class AdaDelta(MinibatchOptimizer):

    epsillon = 10**(-6)
    
    def __init__(self, algorithm, batch_size, decay_rate, verbose=False):
        super().__init__(algorithm, batch_size)

        self.decay_rate = decay_rate
        self.verbose = verbose

        self.running_average = [np.zeros_like(weight) for weight in self.weights]

    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.gradient_function(data_batch, length_batch, label_batch, *self.weights)
        gradients = self.process_gradients(gradients)

        for i, gradient in enumerate(gradients):
            square_gradient = np.square(gradient)
            self.running_average[i] = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * square_gradient

            rmsx = np.sqrt(np.square(self.updates[i]) + self.epsillon)
            rmsgrad = np.sqrt(self.running_average[i] + self.epsillon)
            
            self.updates[i] = -rmsx / rmsgrad * gradient

        if self.verbose:
            print('.', end='', flush=True)

        

class RMSProp(MinibatchOptimizer):

    epsillon = 10**(-6)

    def __init__(self, algorithm, batch_size, decay_rate, learning_rate, verbose=False):
        super().__init__(algorithm, batch_size)

        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.running_average = [np.zeros_like(weight) for weight in self.weights]

    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.gradient_function(data_batch, length_batch, label_batch, *self.weights)
        gradients = self.process_gradients(gradients)
            
        for i, gradient in enumerate(gradients):
            square_gradient = np.square(gradient)
            self.running_average[i] = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * square_gradient

            rmsgrad = np.sqrt(self.running_average[i] + self.epsillon)
            
            self.updates[i] = -self.learning_rate / rmsgrad * gradient

        if self.verbose:
            print('.', end='', flush=True)

        

import sys
import numpy as np

class Optimizer():

    gradient_clipping_factor = None
    use_gradient_noise = False
    gradient_noise_eta = None
    gradient_noise_gamma = None

    def initialize(self):
        pass

    def set_loss_function(self, function):
        self.loss_function = function

    def set_gradient_function(self, function):
        self.gradient_function = function

    def set_initial_weights(self, weights):
        self.weights = weights
        
    def set_update_function(self, function):
        self.update_function = function 

    def do_update(self):
        self.update_function(self.weights)
        
    def set_training_data(self, sentences, labels):
        self.training_sentences = sentences
        self.training_labels = labels
        self.training_lengths = np.array([len(sentence) for sentence in sentences])
        self.training_lengths = self.training_lengths.astype(np.int32)

    def set_development_data(self, sentences, labels):
        self.development_sentences = self.pad_sentences(sentences)
        self.development_labels = self.pad_golds(labels)
        self.development_lengths = np.array([len(sentence) for sentence in sentences])
        self.development_lengths = self.development_lengths.astype(np.int32)
    
    def development_loss(self):
        return self.loss_function(self.development_sentences, self.development_lengths, self.development_labels, *self.weights)
    
    def process_gradients(self, gradients):
        if self.use_gradient_noise:
            std_dev = np.sqrt(self.gradient_noise_eta/(self.current_iteration**self.gradient_noise_gamma))

            gradients = [g + np.random.normal(0, std_dev, size=g.shape) for g in gradients]
        
        if self.gradient_clipping_factor is not None:
            gradient_l2_norm = np.sqrt(sum([np.square(g).sum() for g in gradients]))
            if gradient_l2_norm > self.gradient_clipping_factor:
                gradients = [g*float(self.gradient_clipping_factor)/gradient_l2_norm for g in gradients]

        return gradients


    '''
    Padding:
    '''
        
    def pad_sentences(self, sentence_list):
        longest_sentence = max([len(x) for x  in sentence_list])        
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

    
class MinibatchOptimizer(Optimizer):

    batch_size = None
    max_iterations = None
    
    error_margin = 10**(-8)
    normalize_batches = True

    def initialize(self):
        if self.batch_size is None or self.max_iterations is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()
                        
    def batch_gradients(self, data_batch, length_batch, label_batch):
        aggregate = None
        for sentence, length, gold in zip(data_batch, length_batch, label_batch):
            if aggregate is None:
                aggregate = self.gradient_function(sentence, length, gold, *self.weights)
            else:
                new_g = self.gradient_function(sentence, length, gold, *self.weights)
                aggregate = [aggregate[i] + new_g[i] for i in range(len(new_g))]

        return aggregate
                
        
    def process_gradients(self, gradients):
        if self.normalize_batches:
            gradients = [g/self.batch_size for g in gradients]
        
        gradients = super().process_gradients(gradients)
        return gradients
        
    def chunk(self, l):
        return np.array([l[i:i+self.batch_size] for i in range(0, len(l), self.batch_size)])
        
    def update(self):
        self.current_iteration = 1
        
        length_chunks = self.chunk(self.training_lengths)
        sentence_chunks = self.chunk(self.training_sentences)
        label_chunks = self.chunk(self.training_labels)

        current_loss = self.development_loss()
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

            current_loss = self.development_loss()


class StochasticGradientDescent(MinibatchOptimizer):

    def initialize(self):
        if self.learning_rate is None or self.momentum is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()
        
    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.gradient_function(data_batch, length_batch, label_batch, *self.weights)
        gradients = self.process_gradients(gradients)
        
        for i, gradient in enumerate(gradients):
            self.updates[i] = -self.learning_rate*gradient + self.momentum * self.updates[i]

        if self.verbose:
            print('.', end='', flush=True)


class AdaDelta(MinibatchOptimizer):

    decay_rate = None
    
    epsillon = 10**(-6)
    
    def initialize(self):
        if self.decay_rate is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()
        
        self.running_average = [np.zeros_like(weight) for weight in self.weights]
        
        
    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.batch_gradients(data_batch, length_batch, label_batch)
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

    decay_rate = None
    learning_rate = None
    
    epsillon = 10**(-6)

    def initialize(self):
        if self.decay_rate is None or self.learning_rate is None:
            raise Exception("Optimizer not fully specified in config file!")

        super().initialize()
        
        self.running_average = [np.zeros_like(weight) for weight in self.weights]
        
        
    def batch_update(self, data_batch, length_batch, label_batch):
        gradients = self.batch_gradients(data_batch, length_batch, label_batch)
        gradients = self.process_gradients(gradients)
            
        for i, gradient in enumerate(gradients):
            square_gradient = np.square(gradient)
            self.running_average[i] = self.decay_rate * self.running_average[i] + (1 - self.decay_rate) * square_gradient

            rmsgrad = np.sqrt(self.running_average[i] + self.epsillon)
            
            self.updates[i] = -self.learning_rate / rmsgrad * gradient

        if self.verbose:
            print('.', end='', flush=True)

        
def from_config(config):
    if 'algorithm' not in config:
        raise Exception('Optimization algorithm not specified!')

    if config['algorithm'] == 'SGD':
        optimizer = StochasticGradientDescent()
    elif config['algorithm'] == 'RMSProp':
        optimizer = RMSProp()
    elif config['algorithm'] == 'AdaDelta':
        optimizer = AdaDelta()
    else:
        raise Exception('Optimization algorithm \"'+config['algorithm']+'\" unknown!')

    if 'verbose' in config:
        optimizer.verbose = bool(config['verbose'])

    if 'max iterations' in config:
        optimizer.max_iterations = int(config['max iterations'])
        
    if 'batch size' in config:
        optimizer.batch_size = int(config['batch size'])

    if 'learning rate' in config:
        optimizer.learning_rate = float(config['learning rate'])

    if 'decay rate' in config:
        optimizer.decay_rate = float(config['decay rate'])
        
    if 'gradient clip' in config:
        optimizer.gradient_clipping_factor = float(config['gradient clip'])
    
    if 'gradient noise' in config:
        if 'noise eta' not in config or 'noise gamma' not in config:
            raise Exception('Gradient noise parameters not specified')
        optimizer.use_gradient_noise = bool(config['gradient noise'])
        optimizer.gradient_noise_eta = float(config['noise eta'])
        optimizer.gradient_noise_gamma = float(config['noise gamma'])

    return optimizer

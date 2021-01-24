import numpy as np
import random

class Hopfield:
    def __init__(self, memories):
        ''' Neurons takes binary value 0 or 1.
        args:
            N (int): dimension of the net
            memories (nparray): memory patterns with dimension [number of examples, N]
        returns: None
        '''
        N = memories.shape[1]
        self.weights = np.zeros([N, N])
        self.N = N
        for m in memories:
            self.weights += np.outer(m, m)
        for i in range(N):
            self.weights[i][i] = 0.

        self.neurons = np.zeros(N)

    def set(self, state):
        '''
        Set neuron state of the network
        Args:
            state (list): Neuron state

        Returns: None
        '''
        self.neurons = np.array(state)

    def energy(self, neurons = None):
        '''
        Args:
            neurons: input neurons used to calculate the energy. None for using the network's neuron
        Returns: energy of the network

        '''
        if neurons is None:
            neurons = self.neurons
        return -0.5*float(np.dot(np.matmul(neurons[np.newaxis,:], self.weights), neurons))

    def update(self, num_updates = 1000, rule = 'energy_diff', sync = False):
        ''' Update network
        Args:
            num_updates (int): update of updates to perform
            rule (str): update rule.
                'energy_diff': energy difference
                'field': flip by effective field
        Returns:
        '''

        for n in range(num_updates):
            old_neuron = self.neurons.copy()
            update_order = np.arange(0, self.N)
            random.shuffle(update_order)

            if rule == 'energy_diff':
                for i in update_order:
                        # flip a neuron state if energy is lower after flipping
                        new_neurons = self.neurons.copy()
                        new_neurons[i] *= -1
                        energy_diff = self.energy(new_neurons) - self.energy()
                        if energy_diff < 0:
                            self.neurons[i] = new_neurons[i]
            elif rule == 'field':
                threshold = 0.
                if sync:
                    # update all at the same time
                    self.neurons = (self.field(None) > threshold).astype('float')
                    self.neurons[self.neurons == 0.] = -1
                else:
                    # update one at a time
                    for i in update_order:
                        if self.field(i) > threshold:
                            self.neurons[i] = 1.
                        else:
                            self.neurons[i] = -1
            else:
                raise ValueError('Invalid update rule %s.'%rule)

            if np.array_equal(self.neurons, old_neuron):
                #print('converges after %d updates'%(n+1) )
                return n+1
        return n

    def field(self, i):
        '''return field at the i-th neuron'''
        if i is None:
            return self.weights@(self.neurons)
        return np.dot(self.weights[i], self.neurons)

class ModernHopfield:
    '''As in paper "Dense Associative Memory for Pattern Recognition"  '''
    def __init__(self, memories):
        ''' Note that this model has no weights but stores the training examples (memories) directly.
        args:
            N (int): dimension of the net
            memories (nparray): memory patterns with dimension [number of examples, N]
        returns: None
        '''
        self.memories = memories
        self.N = memories.shape[1]
        self.neurons = np.zeros(self.N)

    def set(self, state):
        '''
        Set neuron state of the network
        Args:
            state (list): Neuron state
        '''
        self.neurons = np.array(state)

    def update(self, num_updates = 1000, rule = 'energy_diff', sync = False):
        ''' Update network
        Args:
            num_updates (int): update of updates to perform
            rule (str): update rule.
                'energy_diff': energy difference
                'field': flip by effective field
        Returns:
        '''

        for n in range(num_updates):
            old_neuron = self.neurons.copy()
            update_order = np.arange(0, self.N)
            random.shuffle(update_order)

            if rule == 'energy_diff':
                for i in update_order:
                        # flip a neuron state if energy is lower after flipping
                        new_neurons = self.neurons.copy()
                        new_neurons[i] *= -1
                        energy_diff = self.energy(new_neurons) - self.energy()
                        if energy_diff < 0:
                            self.neurons[i] = new_neurons[i]
            else:
                raise ValueError('Invalid update rule %s.'%rule)

            if np.array_equal(self.neurons, old_neuron):
                return n+1
        return n

    def energy(self, neurons = None, n = 2):
        '''
        Polynomial energy
        Args:
            neurons: state of neurons, None for the current state
            n: power of norm
        Returns: scalar energy
        '''
        if neurons is None:
            neurons = self.neurons
        x = (np.tile(neurons, (len(self.memories), 1)) * self.memories).sum(1)
        E = - self._F(x, n).sum()
        return E

    def _F(self, x, n):
        '''Rectified polynomial'''
        x[x < 0] = 0.
        return x**n

def addNoise(X, prob = 0.05):
    Xnew = X.copy()
    for i in range(len(Xnew)):
        if random.random() < prob:
            Xnew[i] *= -1
    return Xnew

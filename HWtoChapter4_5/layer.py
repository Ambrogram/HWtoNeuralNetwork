import numpy as np
from HWtoChapter4_5.neuron import Neuron


class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(np.random.rand(num_inputs), 0) for _ in range(num_neurons)]

    def forward(self, inputs, activation_function):
        return [neuron.activate(inputs, activation_function) for neuron in self.neurons]
    

    def backpropagate(self, error, activation_derivative):
        layer_error = np.dot(error, [neuron.weights for neuron in self.neurons])
        gradients = [neuron.compute_gradients(error, activation_derivative) for neuron in self.neurons]
        return layer_error, gradients


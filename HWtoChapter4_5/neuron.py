import numpy as np


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        self.output = 0
        self.input = 0

    def activate(self, inputs, activation_function):
        self.input = np.dot(inputs, self.weights) + self.bias
        self.output = activation_function(self.input)
        return self.output

    def compute_gradients(self, error, activation_derivative):
        return error * activation_derivative(self.output), error * activation_derivative(self.output) * self.input

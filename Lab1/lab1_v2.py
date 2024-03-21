import numpy as np

import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.activation = activation
        self.Z = None  # Linear activation cache
        self.A = None  # Post-activation cache
        self.dW = None  # Gradient of weights
        self.db = None  # Gradient of biases
        self.dA_prev = None  # Gradient of activation

    def forward(self, A_prev):
        self.Z = np.dot(self.weights, A_prev) + self.biases
        self.A = self._apply_activation(self.Z)
        return self.A

    def backward(self, dA, A_prev):
        if self.activation == "relu":
            dZ = dA * self._relu_derivative(self.Z)
        elif self.activation == "sigmoid":
            dZ = dA * self._sigmoid_derivative(self.A)
        else:
            dZ = dA  # Linear or no activation

        m = A_prev.shape[1]
        self.dW = np.dot(dZ, A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        self.dA_prev = np.dot(self.weights.T, dZ)
        return self.dA_prev

    def update_params(self, learning_rate=0.01):
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db

    def _apply_activation(self, Z):
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        return Z  # Linear activation by default

    def _relu_derivative(self, Z):
        return Z > 0

    def _sigmoid_derivative(self, A):
        return A * (1 - A)




class NeuralNetwork:
    def __init__(self, layer_dims, activations):
        self.layers = []
        for i in range(1, len(layer_dims)):
            self.layers.append(Layer(layer_dims[i-1], layer_dims[i], activations[i-1]))
    
    def forward_propagation(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A





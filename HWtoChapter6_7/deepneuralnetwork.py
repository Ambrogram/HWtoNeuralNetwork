import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class DeepNeuralNetwork:
    def __init__(self, X, Y, layer_sizes, alpha=0.1, iterations=500):
        self.X = X
        self.Y = Y
        self.layer_sizes = layer_sizes
        self.alpha = alpha
        self.iterations = iterations
        self.parameters = self.init_params()

    def init_params(self):
        np.random.seed(1)  # Ensure consistent initialization
        params = {}
        layer_count = len(self.layer_sizes)
        for l in range(1, layer_count):
            params['W' + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * 0.01
            params['b' + str(l)] = np.zeros((self.layer_sizes[l], 1))
        return params

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        A = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return A / np.sum(A, axis=0, keepdims=True)

    def forward_prop(self):
        cache = {}
        A = self.X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.ReLU(Z)
            cache['A' + str(l)] = A
            cache['Z' + str(l)] = Z

        # Output layer
        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        Z = np.dot(W, A) + b
        A = self.softmax(Z)
        cache['A' + str(L)] = A
        cache['Z' + str(L)] = Z

        return A, cache

    def compute_cost(self, A2):
        m = self.Y.shape[1]
        cost = -np.sum(np.log(A2) * self.Y) / m
        return cost

    # Include backward_prop and update_params methods here (omitted for brevity)

    def fit(self):
        for i in range(self.iterations):
            A2, cache = self.forward_prop()
            cost = self.compute_cost(A2)
            gradients = self.backward_prop(cache)
            self.update_params(gradients)
            if i % 100 == 0:
                print(f"Iteration {i}, Cost: {cost}")

    # Include methods for predictions and accuracy (omitted for brevity)

# Data loading and preprocessing
data = pd.read_csv('D:/path_to_your_file/train.csv')
data = np.array(data)
np.random.shuffle(data)

# Assuming the dataset structure and normalization as per the provided code
# You can add your data preprocessing steps here

# Example usage
layer_sizes = [784, 128, 10]  # Example layer sizes (input, hidden, output)
nn = DeepNeuralNetwork(X_train, Y_train, layer_sizes)
nn.fit()

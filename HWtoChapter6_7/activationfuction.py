import numpy as np

class ActivationFunctions:
    def linear(self, x):
        """ 
        Linear activation function: f(x) = x
        Purpose: Used where we want the output to be proportional to the input, typically in regression problems 
        or when no activation is required.
        """
        return x

    def relu(self, x):
        """ 
        ReLU activation function: f(x) = max(0, x)
        Purpose: Commonly used in hidden layers of neural networks. It helps with non-linearities 
        and is computationally efficient. It's effective in addressing the vanishing gradient problem.
        """
        return np.maximum(0, x)

    def sigmoid(self, x):
        """ 
        Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
        Purpose: Often used in the output layer for binary classification problems. It maps input values 
        to a range between 0 and 1, modeling probability-like outputs.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """ 
        Tanh activation function: f(x) = tanh(x)
        Purpose: Similar to sigmoid but outputs values from -1 to 1. It’s often used in hidden layers 
        of a neural network as it centers the data, making it easier to learn in subsequent layers.
        """
        return np.tanh(x)

    def softmax(self, x):
        """ 
        Softmax activation function: f(x) = exp(x) / sum(exp(x))
        Purpose: Typically used in the output layer of a multi-class classification network. 
        It converts logits into probabilities by normalizing the output into a probability distribution.
        """
        e_x = np.exp(x - np.max(x)) # subtract max for numerical stability
        return e_x / e_x.sum(axis=0)

# Example of using the class
activations = ActivationFunctions()

# Test data
x = np.array([1, 2, 3])

print("Linear:", activations.linear(x))
print("ReLU:", activations.relu(x))
print("Sigmoid:", activations.sigmoid(x))
print("Tanh:", activations.tanh(x))
print("Softmax:", activations.softmax(x))

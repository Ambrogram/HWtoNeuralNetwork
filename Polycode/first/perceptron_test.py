import numpy as np  # Importing the NumPy library, which provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

def sigmoid(x):
    # Define the sigmoid function, which is an activation function used to map predictions to probabilities.
    return 1 / (1 + np.exp(-x))  # The sigmoid function formula: f(x) = 1 / (1 + e^(-x))

def sigmoid_derivative(x):
    # Define the derivative of the sigmoid function. This is used during the backpropagation step to update weights.
    return x * (1 - x)  # The derivative of the sigmoid function can be expressed as f'(x) = f(x) * (1 - f(x))

# Define the training inputs for the neural network.
# Each sub-array represents a single training example.
# In this case, there are 4 examples with 3 features each.
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# Define the expected outputs for each training example.
# The .T is a transpose, turning the row vector into a column vector.
training_outputs = np.array([[0,1,1,0]]).T

# Initialize the seed for the random number generator to ensure the results are reproducible.
np.random.seed(1)

# Initialize synaptic weights with random values in a 3x1 matrix.
# Weights are initially set to random values between -1 and 1.
synaptic_weights = 2 * np.random.random((3,1)) - 1

# Print the initial, random synaptic weights.
print('Random starting synaptic weights')
print(synaptic_weights)

# Start the training process.
for iteration in range(1000):  # Loop over 1000 iterations (epochs)
    
    # Set the input layer to the training inputs.
    # In this network, the input layer directly receives the training data.
    input_layer = training_inputs
    
    # Calculate the predictions (outputs) by applying the sigmoid function to the weighted sum of inputs.
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    # Compute the error as the difference between actual and predicted outputs.
    error = training_outputs - outputs
    
    # Compute adjustments to the weights based on the error.
    # This uses the derivative of the sigmoid function for gradient descent.
    adjustments = error * sigmoid_derivative(outputs)
    
    # Update synaptic weights.
    # Adjustments are applied to the weights based on the inputs and the error derivative.
    synaptic_weights += np.dot(input_layer.T, adjustments)
    
# After training is complete, print the optimized synaptic weights.
print('Synaptic weights after training')
print(synaptic_weights)
   
# Print the outputs of the network after training.
# These outputs represent the network's predictions based on the training data.
print('Outputs after training:')
print(outputs)

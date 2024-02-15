import numpy as np  # Import the numpy library for numerical operations.

class NeuralNetwork:
    # Define a class for the Neural Network.

    def __init__(self):
        # The initializer method for the NeuralNetwork class.
        np.random.seed(1)  # Set the seed for random number generation for reproducibility.
        
        # Initialize synaptic weights randomly with a 3x1 matrix. Values range from -1 to 1.
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # Define the sigmoid activation function.
        return 1 / (1 + np.exp(-x))  # The sigmoid formula, applied element-wise.

    def sigmoid_derivative(self, x):
        # Define the derivative of the sigmoid function, used in backpropagation.
        return x * (1 - x)  # The derivative of the sigmoid function.

    def train(self, training_inputs, training_outputs, training_iterations):
        # Define the training method for the neural network.
        for _ in range(training_iterations):
            # Loop for the specified number of training iterations.
            
            outputs = self.think(training_inputs)
            # Forward pass: compute the network output based on current synaptic weights.
            
            error = training_outputs - outputs
            # Compute the error as the difference between actual and predicted outputs.
            
            # Calculate adjustments to the weights. This uses the derivative of the sigmoid
            # function for gradient descent optimization.
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(outputs))
            
            self.synaptic_weights += adjustments
            # Update the synaptic weights by adding the calculated adjustments.

    def think(self, inputs):
        # Define the method for a forward pass in the neural network.
        inputs = inputs.astype(float)
        # Convert inputs to float to ensure numerical operations are correctly performed.
        
        # Compute the output of the neural network using the sigmoid activation function.
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        
        return output  # Return the output.

# This block executes when the script is run directly.
if __name__ == "__main__":
    
    neural_network = NeuralNetwork()  # Instantiate the NeuralNetwork class.
    
    print('Random starting synaptic weights')
    print(neural_network.synaptic_weights)  # Print the initial, randomly assigned synaptic weights.
    
    # Define the training data inputs. Each sub-array is a training example.
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    
    # Define the expected outputs for each training example (in a column vector).
    training_outputs = np.array([[0, 1, 1, 0]]).T
    
    neural_network.train(training_inputs, training_outputs, 10000)
    # Train the neural network with the specified training data and iterations.
    
    print('Synaptic weights after training')
    print(neural_network.synaptic_weights)  # Print the synaptic weights after training.
    
    # Get user input for a new situation and convert to float.
    A = float(input('Input 1: '))
    B = float(input('Input 2: '))
    C = float(input('Input 3: '))
    
    print('New situation: input data =', A, B, C)
    # Print the new situation with the provided input data.
    
    # Print the output data predicted by the neural network for the new input.
    print('Output data:', neural_network.think(np.array([A, B, C])))

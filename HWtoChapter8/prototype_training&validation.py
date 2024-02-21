import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Read the data
data = pd.read_csv('D:/Study Files/NEU/Academic/INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]/train.csv')
data = np.array(data)  # Convert data to a NumPy array
np.random.shuffle(data)  # Shuffle the data

# Splitting data into training, validation, and testing sets
m, n = data.shape
train_end = int(m * 0.70)  # 70% of the data
valid_end = int(m * 0.85)  # Additional 15% of the data

data_train = data[:train_end].T
data_valid = data[train_end:valid_end].T
data_test = data[valid_end:].T  # Remaining 15% of the data

# Extracting labels and features for each set and normalizing features
Y_train, X_train = data_train[0], data_train[1:n] / 255.
Y_valid, X_valid = data_valid[0], data_valid[1:n] / 255.
Y_test, X_test = data_test[0], data_test[1:n] / 255.

#  now proceed with the rest of your neural network code...
# ... (including function definitions and training procedures)

def init_params():
    # Initializes weights and biases for two layers of a neural network.
    # W1 and W2 are weight matrices, and b1 and b2 are bias vectors.
    
    # W1 is initialized with random values for connections between the input layer (784 units, e.g., for an image input like MNIST dataset) 
    # and the first hidden layer (10 units). Random initialization helps to break symmetry in the learning process.
    # Subtracting 0.5 centers the values around zero.

    """
    The numbers in the brackets (10, 784) in the line W1 = np.random.rand(10, 784) - 0.5 specify the shape of the array that is being created and filled with random values. Let's break down what each number represents: 10: 
    This number represents the number of neurons (or units) in the first hidden layer of the neural network. In the context of this code, it means that there are 10 neurons in the first hidden layer.

    784: This number corresponds to the number of input features. In the example given, it is suggested that the network is designed for an input like the MNIST dataset, where each input is an image of 28x28 pixels. Since 28x28 equals 784, 
    each image is represented as a flattened array of 784 features (each feature representing one pixel's intensity).

    So, when np.random.rand(10, 784) is executed, it creates a 2-dimensional NumPy array with the shape of 10 rows and 784 columns. Each element of this array is a randomly generated number between 0 and 1 (as np.random.rand generates random 
    numbers in this range). Subtracting 0.5 from each element shifts these random values to be centered around 0, ranging from -0.5 to 0.5. This centering is a common practice in neural network initialization as it can help in speeding up the
    convergence of the training process.

    In the context of a neural network:

    Each row of the array W1 represents the weights connecting all input features to a particular neuron in the first hidden layer.
    Each column corresponds to a specific input feature's weight across all neurons in the layer.
    This arrangement of weights is fundamental for the matrix operations performed during the forward pass of a neural network. The weights in W1 will be updated during the training process to minimize the loss function of the network.
    """
    
        
    W1 = np.random.rand(10, 784) - 0.5
    
    # b1 is the bias vector for the first hidden layer, initialized similarly to W1.
    b1 = np.random.rand(10, 1) - 0.5
    
    # W2 is the weight matrix between the first hidden layer and the output layer (both having 10 units).
    # Initialized randomly to maintain diversity in the initial learned representations.
    """
    First 10: This number represents the number of neurons in the layer that W2 is associated with. In this context, it indicates that W2 is the weight matrix for a layer with 10 neurons.

    Second 10: This second number represents the number of neurons in the preceding layer (or the number of inputs to the layer for which W2 is the weight matrix). Since it is also 10, 
    it implies that the previous layer (which could be either an input layer or another hidden layer) also consists of 10 neurons.

    When np.random.rand(10, 10) is executed, it creates a 2-dimensional NumPy array with 10 rows and 10 columns. Each element of this array is a randomly generated number between 0 and 1. 
    Subtracting 0.5 from these values shifts the range of these initial weights to be between -0.5 and 0.5. This range is often preferred in neural network initialization as it helps in 
    maintaining a balance in the initial neuron activations, preventing them from starting too large or too small.

    In the Context of a Neural Network:
    Each row of W2 corresponds to one neuron in the layer and contains the weights connecting that neuron to all neurons in the previous layer. For example, the first row in W2 contains 
    the weights that connect the first neuron in this layer to all 10 neurons in the previous layer.
    Each column of W2 represents the weights associated with a particular neuron in the previous layer as they connect to every neuron in the current layer. For instance, the first column 
    in W2 represents the weights from the first neuron in the previous layer to each of the 10 neurons in the current layer.
    An Example:
    If this is a neural network for image classification, W2 might be the weights between two hidden layers, both containing 10 neurons. The weight W2[i, j] would represent the strength of 
    the connection from the j-th neuron in the first hidden layer to the i-th neuron in the second hidden layer. These weights are crucial as they define how the activations from one layer 
    propagate to the next, and they will be iteratively adjusted during the training process.
    """
    
    W2 = np.random.rand(10, 10) - 0.5
    
    # b2 is the bias vector for the output layer.
    b2 = np.random.rand(10, 1) - 0.5
    
    # The function returns the initialized parameters.
    return W1, b1, W2, b2

def ReLU(Z):
    # ReLU (Rectified Linear Unit) activation function.
    # It returns the element-wise maximum of the input 'Z' and 0.
    # This function introduces non-linearity to the model, allowing it to learn more complex functions.
    return np.maximum(Z, 0)

def softmax(Z):
    # Softmax function, commonly used as the activation function in the output layer of a neural network for classification.
    # It converts the input 'Z' into a probability distribution, where each value lies between 0 and 1, and the sum of all values is 1.
    # 'np.exp(Z)' calculates the exponential of each element in 'Z'.
    # The result is normalized by dividing by the sum of these exponentials, ensuring a valid probability distribution.
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    # Forward propagation function computes the predictions of the neural network.
    # 'X' is the input data, 'W1', 'b1', 'W2', and 'b2' are the network parameters.
    
    # First layer forward pass: Computes the linear combination of inputs and weights, adds the bias (Z1),
    # and then applies the ReLU activation function.
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    # Second layer forward pass: Similar to the first layer but uses the output of the first layer 'A1' as input.
    # The softmax function is applied to the output, producing a probability distribution as the final output.
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
    # Returns the intermediate values for use in backpropagation.
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    # Derivative of the ReLU function.
    # Used in backpropagation to compute gradients. For ReLU, the derivative is 1 if Z is greater than 0, else it's 0.
    # Z is an array of values for which the derivative of ReLU needs to be computed.
    # The expression 'Z > 0' performs an element-wise comparison of each element in Z with 0.
    # It returns an array of the same shape as Z, where each element is 'True' if the corresponding element in Z is greater than 0, and 'False' otherwise.
    # In Python, 'True' is interpreted as 1 and 'False' as 0 when used in numerical contexts.
    
    """
    Derivative of ReLU: The derivative of a function gives us the slope of the function at any point. For the ReLU function:

    When x > 0, the slope is 1 (since the function is just y = x in this region, which is a straight line with a slope of 1).
    When x <= 0, the slope is 0 (the function is constant at y = 0 in this region).
    Z > 0 Expression:

    Z is an array, and Z > 0 performs an element-wise comparison of each element in Z with 0.
    The result is a boolean array where each element is True if the corresponding element in Z is greater than 0, and False otherwise.
    Interpretation in Python:

    In Python, True is treated as 1 and False as 0 in numerical contexts. So, when the boolean array resulting from Z > 0 is used in calculations, it effectively becomes an array of 1s and 0s.
    Therefore, return Z > 0 is a concise way of saying "return 1 where Z is positive and 0 elsewhere," which is exactly the derivative of ReLU.
    Example:
    """
    return Z > 0

def one_hot(Y):
    # Converts a vector of labels 'Y' into a one-hot encoded matrix.
    # One-hot encoding is used to convert categorical labels into a format suitable for multi-class classification.
    
    # Creates a matrix of zeros with a shape (number of classes, number of samples).
    # 'Y.size' is the number of samples, and 'Y.max() + 1' provides the number of classes.
    
    # 'Y' is a 1D array of categorical labels. These labels are integers (e.g., class 0, class 1, class 2, etc.).
    
    # Initialize a matrix of zeros. The shape is (number of samples, number of classes).
    # 'Y.size' gives the total number of samples. 'Y.max() + 1' is used to determine the number of unique classes.
    # The '+1' is necessary because class labels typically start from 0. 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    
    # Sets the appropriate elements to 1 to complete the one-hot encoding.
    # 'np.arange(Y.size)' creates an array with indices for each sample,
    # 'Y' contains the class labels. This combination is used to index 'one_hot_Y'.
    
    
    one_hot_Y[np.arange(Y.size), Y] = 1
    
    # Transpose the matrix to align with the expected layout (classes as rows, samples as columns).
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # Backward propagation function computes the gradients of the loss function with respect to the parameters.
    # This is crucial for updating the parameters in the direction that minimizes the loss.
    
    # Convert labels 'Y' into one-hot encoded format.
    one_hot_Y = one_hot(Y)
    
    # Calculate the derivative of the cost with respect to Z2 (output of the last layer).
    # 'A2 - one_hot_Y' is the difference between predicted probabilities and actual values (one-hot encoded).
    dZ2 = A2 - one_hot_Y
    
    # Compute the gradient of the cost with respect to the weights of the second layer.
    # '1/m' normalizes the gradient by the number of samples.
    dW2 = 1 / m * dZ2.dot(A1.T)
    
    # Compute the gradient of the cost with respect to the bias of the second layer.
    db2 = 1 / m * np.sum(dZ2)
    
    # Compute the derivative of the cost with respect to Z1 (output of the first layer).
    # It involves the derivative of ReLU and a dot product with the transposed weights of the second layer.
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    
    # Compute gradients with respect to the weights and bias of the first layer, similar to the second layer.
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    # Return the gradients for updating the parameters.
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Update the parameters of the neural network using the gradients computed in backpropagation.
    # 'alpha' is the learning rate, which controls the size of the update steps.
    
    # Update weights and biases by subtracting the product of the learning rate and the respective gradients.
    # This step moves the parameters in the direction that minimizes the loss.
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    
    # Return the updated parameters.
    return W1, b1, W2, b2


def get_predictions(A2):
        # Returns the index of the highest value in each column of array A2.
        # A2 is expected to be a 2D array where each column represents a different sample and each row represents a different class.
        # np.argmax(A2, 0) finds the indices of the maximum values along the first axis (rows) for each column.
        return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    # Calculates the accuracy of the predictions.
    # 'predictions' is a 1D array of predicted class labels.
    # 'Y' is a 1D array of actual class labels.
    # The expression 'predictions == Y' performs an element-wise comparison, resulting in a boolean array.
    # 'np.sum(predictions == Y)' counts the number of True values (correct predictions).
    # Dividing by 'Y.size' (total number of samples) gives the proportion of correct predictions, which is the accuracy.
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    # Performs the gradient descent optimization algorithm.
    # 'X' and 'Y' are the input features and labels, respectively.
    # 'alpha' is the learning rate, a hyperparameter that controls the step size in each iteration.
    # 'iterations' is the number of times the algorithm will update the parameters.

    # Initialize weights and biases.
    W1, b1, W2, b2 = init_params()

    # Iterate for the given number of iterations.
    for i in range(iterations):
        # Perform forward propagation to calculate current predictions.
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # Perform backward propagation to calculate the gradients.
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # Update the parameters using the calculated gradients and the learning rate.
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Every 10 iterations, print the iteration number and the current accuracy.
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))

    # Return the updated parameters after all iterations.
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

def make_predictions(X, W1, b1, W2, b2):
    # Conducts forward propagation through the neural network to make predictions based on input X.
    
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    # Calls the forward_prop function, passing in the weights, biases, and input data.
    # The forward_prop function returns four values (Z1, A1, Z2, A2), but only A2 (the activation of the last layer) is needed here.
    # A2 contains the network's output for the input data X.

    predictions = get_predictions(A2)
    # Converts the output of the neural network (A2) into class predictions.
    # It uses the get_predictions function, which typically applies np.argmax to find the index of the maximum value in each column of A2.
    # This index represents the predicted class.

    return predictions
    # Returns the predicted classes for the input data X.

def test_prediction(index, W1, b1, W2, b2):
    # Tests and visualizes the prediction for a specific sample in the dataset.
    
    current_image = X_train[:, index, None]
    # Extracts the image data for the given 'index' from the training set X_train.
    # The 'None' is used to maintain a 2D shape, which is necessary for matrix operations in the neural network.

    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    # Makes a prediction using the make_predictions function for the selected image.
    # The same weights and biases from the trained network are used.

    label = Y_train[index]
    # Retrieves the actual label for the selected image from the training labels Y_train.

    print("Prediction: ", prediction)
    print("Label: ", label)
    # Prints out the predicted class and the actual label for comparison.

    current_image = current_image.reshape((28, 28)) * 255
    # Reshapes the image data from a flattened array back into a 28x28 format suitable for displaying.
    # Multiplies by 255 to convert the normalized pixel values back to the original scale (0-255).

    plt.gray()
    # Sets the color map to grayscale for displaying the image.

    plt.imshow(current_image, interpolation='nearest')
    # Displays the image using matplotlib's imshow function with nearest-neighbor interpolation.

    plt.show()
    # Renders the plot, showing the image.
    
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# Example: Use gradient_descent to train on the training set
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)

# After training,  evaluate your model's performance on the validation and test sets
# Example: Evaluate on the validation set
predictions_valid = make_predictions(X_valid, W1, b1, W2, b2)
valid_accuracy = get_accuracy(predictions_valid, Y_valid)
print("Validation Accuracy:", valid_accuracy)

# Example: Evaluate on the test set
predictions_test = make_predictions(X_test, W1, b1, W2, b2)
test_accuracy = get_accuracy(predictions_test, Y_test)
print("Test Accuracy:", test_accuracy)

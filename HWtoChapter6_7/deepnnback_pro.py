import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
data = pd.read_csv('D:/Study Files/NEU/Academic/INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]/train.csv')



# Convert 'data' from its original format (which could be a list, a Pandas DataFrame, or another format) into a NumPy array.
# NumPy arrays are a central data structure of the NumPy library. They enable efficient operations on large multi-dimensional arrays,
# which are often required in scientific computing and machine learning tasks.
# This conversion standardizes the data format, simplifying subsequent operations like mathematical computations.
data = np.array(data)

# 'm' and 'n' are variables used to store the dimensions of the array 'data'.
# Here, 'm' represents the number of data points (or samples) in the dataset, and 'n' represents the number of features (or attributes) each data point has.
# 'data.shape' returns a tuple where the first element is the number of rows (representing the data points),
# and the second element is the number of columns (representing the features).
# Understanding the shape of the data is crucial in machine learning as it affects how data is handled and fed into models.
m, n = data.shape

# Shuffle the data randomly. This is an important step in many machine learning pipelines to ensure that
# any order or pattern present in the dataset (which could be a result of how the data was collected or organized)
# does not bias the training process. Random shuffling helps in creating a more generalized model.
np.random.shuffle(data)

# Split the data into two sets - a development (or validation) set and a training set.
# The development set is used to tune hyperparameters and evaluate the performance of the model during and after training,
# while the training set is used to train the model.
# Here, the first 1000 samples of the shuffled data are allocated to the development set.
# Transposing the array with '.T' changes its shape from (samples, features) to (features, samples),
# which is often required for machine learning libraries where each column represents a different data point.
data_dev = data[0:1000].T

# Extract the labels (Y) for the development set.
# It's assumed that in this dataset, the first row (or column after transposing) contains the labels of the samples.
# Labels are what the model will try to predict; they are separated from the features.
Y_dev = data_dev[0]

# Extract the features (X) for the development set.
# This includes all the data except the labels. '1:n' ensures that all columns (features) except the first one (labels) are selected.
X_dev = data_dev[1:n]

# Normalize the feature values of the development set.
# Normalization is a standard preprocessing step to ensure that the values of different features are on a similar scale,
# often between 0 and 1. This improves the convergence behavior of many machine learning algorithms.
# In this case, it's assumed that the feature values range from 0 to 255 (a common range for image data), so they're scaled down by dividing by 255.
X_dev = X_dev / 255.

# Prepare the training set using the remaining data.
# Similar to the development set, the data is transposed, and the same structure is followed: separating labels and features.
data_train = data[1000:m].T

# Extract the labels (Y) for the training set.
Y_train = data_train[0]

# Extract the features (X) for the training set.
X_train = data_train[1:n]

# Normalize the feature values of the training set.
# As with the development set, this normalization is crucial for most machine learning models to perform effectively.
X_train = X_train / 255.

# Determine the number of training examples after the split.
_, m_train = X_train.shape


def init_params():
    # Initialize parameters for a 10-layer neural network
    layer_sizes = [784] + [64] * 8 + [10]  # Example: 8 hidden layers of size 64, input layer 784, output layer 10
    params = {}
    for l in range(len(layer_sizes)-1):
        params['W' + str(l+1)] = np.random.randn(layer_sizes[l+1], layer_sizes[l]) * 0.1
        params['b' + str(l+1)] = np.zeros((layer_sizes[l+1], 1))
    return params

def forward_prop(params, X):
    # Forward propagation for a 10-layer network
    A = X
    cache = {'A0': X}
    L = len(params) // 2  # Number of layers

    for l in range(1, L):
        Z = params['W' + str(l)].dot(A) + params['b' + str(l)]
        A = ReLU(Z)
        cache['A' + str(l)] = A
        cache['Z' + str(l)] = Z

    # Output layer
    Z = params['W' + str(L)].dot(A) + params['b' + str(L)]
    A = softmax(Z)
    cache['A' + str(L)] = A
    cache['Z' + str(L)] = Z

    return cache

def backward_prop(cache, params, X, Y):
    # Backward propagation for a 10-layer network
    gradients = {}
    L = len(params) // 2
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    dZ = cache['A' + str(L)] - one_hot_Y
    for l in reversed(range(1, L + 1)):
        gradients['dW' + str(l)] = 1/m * dZ.dot(cache['A' + str(l-1)].T)
        gradients['db' + str(l)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dZ = params['W' + str(l)].T.dot(dZ) * ReLU_deriv(cache['Z' + str(l-1)])

    return gradients

# Update gradient_descent, make_predictions, and test_prediction functions to use these new forward_prop and backward_prop
# Other functions like ReLU, softmax, ReLU_deriv, one_hot, get_predictions, get_accuracy remain the same

# Example of using the updated network
params = init_params()
# Use gradient_descent function to train the network
# Use make_predictions and test_prediction to make and test predictions


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


def forward_prop(params, X):
    # Forward propagation for a 10-layer network
    # Initialize the input data as the first set of activations
    A = X
    cache = {'A0': X}
     # Calculate the number of layers in the network based on the parameters
    L = len(params) // 2  # Number of layers

    # Iterate over each layer, excluding the output layer
    for l in range(1, L):
        # Compute the linear part of the neuron's computation for layer l
        Z = params['W' + str(l)].dot(A) + params['b' + str(l)]
        # Apply the ReLU activation function to get the activations for layer l
        A = ReLU(Z)
        # Store the activations and pre-activation values for layer l in the cache
        cache['A' + str(l)] = A
        cache['Z' + str(l)] = Z

    # Handling the output layer
    # Compute the linear part of the neuron's computation for the output layer
    Z = params['W' + str(L)].dot(A) + params['b' + str(L)]
    # Apply the softmax function to get the activations for the output layer
    A = softmax(Z)
    # Store the activations and pre-activation values for the output layer in the cache
    cache['A' + str(L)] = A
    cache['Z' + str(L)] = Z

    # Return the cache containing all activations and pre-activation values
    return cache


def backward_prop(cache, params, X, Y):
    # Backward propagation for a 10-layer network
    gradients = {}
    L = len(params) // 2
    m = X.shape[1]
    one_hot_Y = one_hot(Y)

    dZ = cache['A' + str(L)] - one_hot_Y
    for l in reversed(range(1, L + 1)):
        gradients['dW' + str(l)] = 1/m * dZ.dot(cache['A' + str(l-1)].T)
        gradients['db' + str(l)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            dZ = params['W' + str(l)].T.dot(dZ) * ReLU_deriv(cache['Z' + str(l-1)])

    return gradients


def update_params(params, gradients, alpha):
    L = len(params) // 2  # Number of layers
    for l in range(1, L + 1):
        params['W' + str(l)] -= alpha * gradients['dW' + str(l)]
        params['b' + str(l)] -= alpha * gradients['db' + str(l)]
    return params



def compute_cost(A_last, Y):
    m = Y.size  # Changed from Y.shape[1] to Y.size for 1D array
    logprobs = np.log(A_last[Y, np.arange(m)])
    cost = -np.sum(logprobs) / m
    return cost





def gradient_descent(X, Y, alpha, iterations):
    params = init_params()
    for i in range(iterations):
        cache = forward_prop(params, X)
        gradients = backward_prop(cache, params, X, Y)
        params = update_params(params, gradients, alpha)

        if i % 10 == 0:  # Example: Print cost every 10 iterations
            A_last = cache['A' + str(len(params) // 2)]
            cost = compute_cost(A_last, Y)
            print(f"Iteration {i}, Cost: {cost}")
    
    return params


def make_predictions(X, params):
    cache = forward_prop(params, X)
    A_last = cache['A' + str(len(params) // 2)]
    predictions = get_predictions(A_last)
    return predictions

def get_predictions(A):
    return np.argmax(A, 0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)


def test_prediction(index, X, Y, params):
    current_image = X[:, index, None]
    prediction = make_predictions(current_image, params)
    label = Y[index]

    print("Prediction: ", prediction)
    print("Label: ", label)

    if current_image.shape[0] == 784:  # Assuming MNIST
        plt.imshow(current_image.reshape(28, 28) * 255, cmap='gray')
        plt.show()


# Assuming you have already loaded and preprocessed your data into X_train and Y_train
alpha = 0.1
iterations = 1000
trained_params = gradient_descent(X_train, Y_train, alpha, iterations)

# Making predictions
predictions = make_predictions(X_dev, trained_params)
print("Development Set Accuracy:", get_accuracy(predictions, Y_dev))

# Testing with an example
test_prediction(0, X_train, Y_train, trained_params)  # Test with the first training example

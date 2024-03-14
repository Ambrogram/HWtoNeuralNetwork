import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load data
data = pd.read_csv('D:/Study Files/NEU/Academic/INFO7375 41272 ST Neural Networks & AI SEC 30 Spring 2024 [OAK-2-LC]/train.csv')
data = np.array(data)
np.random.shuffle(data)  # Shuffle the data

# Split the data
m, n = data.shape
data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.  # Normalize
data_train = data[1000:].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.  # Normalize

# Define neural network architecture
def init_params():
    np.random.seed(1)  # Seed for reproducibility
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(8, 10) * 0.01
    b2 = np.zeros((8, 1))
    W3 = np.random.randn(8, 8) * 0.01
    b3 = np.zeros((8, 1))
    W4 = np.random.randn(4, 8) * 0.01
    b4 = np.zeros((4, 1))
    W5 = np.random.randn(1, 4) * 0.01
    b5 = np.zeros((1, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5}

# Activation functions and their derivatives
def relu(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_deriv(A):
    return A * (1 - A)

# Forward propagation
def forward_propagation(X, params):
    # Retrieve parameters
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    W4, b4 = params['W4'], params['b4']
    W5, b5 = params['W5'], params['b5']

    # First layer
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # Second layer
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # Third layer (repeated architecture of the second layer)
    Z3 = np.dot(W3, A2) + b3
    A3 = relu(Z3)

    # Fourth layer
    Z4 = np.dot(W4, A3) + b4
    A4 = relu(Z4)

    # Output layer
    Z5 = np.dot(W5, A4) + b5
    A5 = sigmoid(Z5)

    # Cache the values for backward propagation
    cache = {
        "Z1": Z1, "A1": A1,
        "Z2": Z2, "A2": A2,
        "Z3": Z3, "A3": A3,
        "Z4": Z4, "A4": A4,
        "Z5": Z5, "A5": A5
    }

    return A5, cache


# Compute cost
def compute_cost(A5, Y):
    """
    Compute the binary cross-entropy cost.
    
    Parameters:
    - A5: The sigmoid output of the last layer, which are the predictions (probabilities)
         that the network has made for each example.
    - Y: The true "label" vector (containing 0 if non-cat, 1 if cat in binary classification).

    Returns:
    - cost: The cross-entropy cost given the predictions A5 and the true labels Y.
    """
    
    m = Y.shape[1]  # Number of examples

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A5), Y) + np.multiply((1 - Y), np.log(1 - A5))
    cost = - np.sum(logprobs) / m
    
    # To ensure cost is not a complex number (which could happen due to numerical instability)
    cost = np.squeeze(cost)  # Makes sure cost is the dimension we expect. 
                             # E.g., turns [[17]] into 17 
    
    assert(cost.shape == ())  # Cost should be a scalar value
    
    return cost


# Backward propagation
def backward_propagation(X, Y, cache, params):
    """
    Implement the backward propagation for the [LINEAR->RELU]*(L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    X -- input data: shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    cache -- cache containing results of forward_propagation(), contains (Z1, A1, Z2, A2, ..., Z5, A5)
    params -- dictionary containing parameters "W1", "b1", ..., "W5", "b5"
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)], grads["dW" + str(l)], grads["db" + str(l)] 
    """
    m = X.shape[1]
    Z1, A1, Z2, A2, Z3, A3, Z4, A4, Z5, A5 = cache['Z1'], cache['A1'], cache['Z2'], cache['A2'], cache['Z3'], cache['A3'], cache['Z4'], cache['A4'], cache['Z5'], cache['A5']
    W1, W2, W3, W4, W5 = params['W1'], params['W2'], params['W3'], params['W4'], params['W5']
    
    # Backward propagation: calculate dW1, db1, dW2, db2, ..., dW5, db5
    dZ5 = A5 - Y
    dW5 = 1./m * np.dot(dZ5, A4.T)
    db5 = 1./m * np.sum(dZ5, axis=1, keepdims=True)
    
    dA4 = np.dot(W5.T, dZ5)
    dZ4 = np.multiply(dA4, relu_deriv(Z4))
    dW4 = 1./m * np.dot(dZ4, A3.T)
    db4 = 1./m * np.sum(dZ4, axis=1, keepdims=True)
    
    dA3 = np.dot(W4.T, dZ4)
    dZ3 = np.multiply(dA3, relu_deriv(Z3))
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, relu_deriv(Z2))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, relu_deriv(Z1))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW5": dW5, "db5": db5,
             "dW4": dW4, "db4": db4,
             "dW3": dW3, "db3": db3,
             "dW2": dW2, "db2": db2,
             "dW1": dW1, "db1": db1}
             
    return grads


# Update parameters
def update_params(params, grads, alpha):
    """
    Update parameters using gradient descent.
    
    Arguments:
    params -- python dictionary containing your parameters 
              params['W' + str(l)] = Wl
              params['b' + str(l)] = bl
    grads -- python dictionary containing your gradients, output of backward_propagation
             grads['dW' + str(l)] = dWl
             grads['db' + str(l)] = dbl
    alpha -- the learning rate, scalar.
    
    Returns:
    params -- python dictionary containing your updated parameters 
    """
    
    L = len(params) // 2  # Number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - alpha * grads["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - alpha * grads["db" + str(l+1)]
        
    return params


# Combine it into a training loop
def model(X, Y, alpha, iterations):
    params = init_params()
    for i in range(iterations):
        A5, cache = forward_propagation(X, params)
        cost = compute_cost(A5, Y)
        grads = backward_propagation(X, Y, cache, params)
        params = update_params(params, grads, alpha)
        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")
    return params

# Now, you can call model(X_train, Y_train, 0.01, 1000) to train your model

# Adjust the code snippets above to complete the missing implementations.
# Note: This code outline omits some details for brevity. You'll need to fill in the forward_propagation, compute_cost,
# backward_propagation, update_params, and potentially other helper functions (e.g., for creating mini-batches) based on the 
# detailed implementations provided earlier and the requirements of your specific problem and dataset.


train_size = 1000  # Define the value of train_size

def get_predictions(A_last):
    return A_last > 0.5

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y) * 100

def make_predictions(X, params):
    A_last, _ = forward_propagation(X, params)
    predictions = get_predictions(A_last)
    return predictions

def test_prediction(X, Y, params):
    predictions = make_predictions(X, params)
    return get_accuracy(predictions, Y)


data_test = data[train_size:].T
Y_test = data_test[0]
X_test = data_test[1:] / 255.  # Normalize

params = model(X_train, Y_train, 0.01, 1000)  # Train the model
A5, _ = forward_propagation(X_test, params)  # Forward propagate the test set




test_prediction(X_test, Y_test, params) 


# Now, you can call test_prediction(X_test, Y_test, params) to evaluate your model on the test set

# Adjust the code snippets above to complete the missing implementations.
# Note: This code outline omits some details for brevity. You'll need to fill in the forward_propagation, compute_cost,

# backward_propagation, update_params, and potentially other helper functions (e.g., for creating mini-batches) based on the
# detailed implementations provided earlier and the requirements of your specific problem and dataset.
# You'll also need to implement the get_predictions, get_accuracy, and make_predictions functions based on the detailed

# implementations provided earlier and the requirements of your specific problem and dataset.

    


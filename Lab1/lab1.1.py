import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


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



import numpy as np

# Activation Functions and Derivatives
class Activation:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def relu_deriv(a):
        return a > 0
    
    @staticmethod
    def sigmoid_deriv(a):
        return a * (1 - a)

# Loss Function
class LossFunction:
    @staticmethod
    def binary_cross_entropy(predicted, true):
        return -np.mean(true * np.log(predicted) + (1 - true) * np.log(1 - predicted))
    
    @staticmethod
    def binary_cross_entropy_deriv(predicted, true):
        return (predicted - true) / (predicted * (1 - predicted))
    


# Layer with Dropout
class Layer:
    def __init__(self, n_input, n_neurons, activation=None, dropout_rate=0.0):
        self.weights = np.random.randn(n_neurons, n_input) * 0.01
        self.biases = np.zeros((n_neurons, 1))
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.Z = None
        self.A = None
        self.D = None  # Dropout mask
        self.dW = None
        self.db = None

    def forward(self, X, training=True):
        # Validate input dimensions:
        if X.shape[0] != self.weights.shape[1]:
            raise ValueError(f"Input shape {X.shape} does not match expected shape ({self.weights.shape[1]}, n_samples)")
        
        self.Z = np.dot(self.weights, X) + self.biases
        
        self.Z = np.dot(self.weights, X) + self.biases
        self.A = self._apply_activation(self.Z)
        
        if training and self.dropout_rate > 0:
            self.D = np.random.rand(*self.A.shape) > self.dropout_rate
            self.A *= self.D
            self.A /= (1 - self.dropout_rate)
        return self.A
    
    def _apply_activation(self, Z):
        if self.activation == 'relu':
            return Activation.relu(Z)
        elif self.activation == 'sigmoid':
            return Activation.sigmoid(Z)
        return Z

    def backward(self, dA, W_next, Z_prev, is_output_layer=False):
        if self.dropout_rate > 0:
            dA *= self.D
            dA /= (1 - self.dropout_rate)
            
        if self.activation == 'relu':
            dZ = np.multiply(dA, Activation.relu_deriv(self.Z))
        elif self.activation == 'sigmoid':
            dZ = np.multiply(dA, Activation.sigmoid_deriv(self.A))
        else:
            dZ = dA
        
        m = Z_prev.shape[1]
        self.dW = np.dot(dZ, Z_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        if not is_output_layer:
            dA_prev = np.dot(W_next.T, dZ)
        else:
            dA_prev = None
        return dA_prev

    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db

# Model with regularization and dropout
class Model:
    def __init__(self, lambda_val=0.0):
        self.layers = []
        self.lambda_val = lambda_val  # For L2 Regularization

    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def compute_loss(self, predicted, true):
        loss = LossFunction.binary_cross_entropy(predicted, true)
        # Add L2 regularization term
        if self.lambda_val > 0:
            l2_cost = 0
            for layer in self.layers:
                l2_cost += np.sum(np.square(layer.weights))
            loss += (self.lambda_val / (2 * predicted.shape[1])) * l2_cost
        return loss

    def backward(self, predicted, true):
        dA = LossFunction.binary_cross_entropy_deriv(predicted, true)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            prev_layer_A = self.layers[i-1].A if i > 0 else None
            dA = layer.backward(dA, self.layers[i+1].weights if i < len(self.layers)-1 else None, prev_layer_A if prev_layer_A is not None else predicted, is_output_layer=(i==len(self.layers)-1))

    def update_params(self, learning_rate):
        for layer in self.layers:
            layer.update_params(learning_rate)
    
    def train(self, X_train, y_train, epochs, learning_rate, batch_size=64, training=True):
        lr = learning_rate
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            y_train_shuffled = y_train[:, permutation]

            for i in range(0, X_train.shape[1], batch_size):
                X_batch = X_train_shuffled[:, i:i+batch_size]
                Y_batch = y_train_shuffled[:, i:i+batch_size]
                
                # Forward pass
                predicted = self.forward(X_batch, training=training)
                
                # Compute loss and perform backward pass
                loss = self.compute_loss(predicted, Y_batch)
                self.backward(predicted, Y_batch)
                
                # Parameter update
                self.update_params(lr)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            # Example learning rate scheduler: Reduce learning rate by half every 500 epochs
            if (epoch+1) % 500 == 0:
                lr /= 2
                print(f"Reduced learning rate to {lr}")



# This setup now includes dropout correctly for training vs. evaluation modes and incorporates L2 regularization.
# Real-world scenarios would require more robust error handling, support for various optimization algorithms like Adam, and evaluation methods.


model = Model()
model.add(Layer(n_input=784, n_neurons=10, activation='relu'))
model.add(Layer(n_input=10, n_neurons=8, activation='relu'))
model.add(Layer(n_input=8, n_neurons=8, activation='relu'))
model.add(Layer(n_input=8, n_neurons=4, activation='relu'))
model.add(Layer(n_input=4, n_neurons=1, activation='sigmoid'))


def get_predictions(A_last):
    """
    Generates binary predictions from the probability outputs of the model.
    """
    return A_last > 0.5

def get_accuracy(predictions, Y):
    """
    Calculates the accuracy of predictions against the true labels.
    """
    return np.mean(predictions == Y) * 100

def make_predictions(X, model):
    """
    Uses the model to make predictions on a given set of inputs.
    """
    A_last = model.forward(X, training=False)  # Set training to False to disable dropout
    predictions = get_predictions(A_last)
    return predictions

def test_prediction(X_test, Y_test, model):
    """
    Tests the model on a given test set and returns the accuracy.
    """
    predictions = make_predictions(X_test, model)
    accuracy = get_accuracy(predictions, Y_test)
    return accuracy





def evaluate_model(X_test, Y_test, model):
    predictions = make_predictions(X_test, model)
    accuracy = get_accuracy(predictions, Y_test)
    precision = precision_score(Y_test.T, predictions.T)
    recall = recall_score(Y_test.T, predictions.T)
    f1 = f1_score(Y_test.T, predictions.T)
    
    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Assuming X_dev and Y_dev are your development set
evaluate_model(X_dev.T, Y_dev.reshape(1, -1), model)



# Assuming X_train and Y_train are loaded and preprocessed correctly
# Ensure Y_train is a 2D row vector
if Y_train.ndim == 1:
    Y_train = Y_train.reshape(1, -1)

# Train the model
model.train(X_train.T, Y_train, epochs=1000, learning_rate=0.01)



# Forward pass on the development set
output_dev = model.forward(X_dev.T, training=False)  # Ensure training=False to disable dropout
predictions_dev = get_predictions(output_dev)

# Calculate and print accuracy
accuracy_dev = get_accuracy(predictions_dev, Y_dev.reshape(1, -1))
print(f"Development Set Accuracy: {accuracy_dev}%")

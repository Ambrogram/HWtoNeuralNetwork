import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load and preprocess data
# Make sure to update the path as per your setup
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

class LossFunction:
    @staticmethod
    def binary_cross_entropy(predicted, true):
        return -np.mean(true * np.log(predicted) + (1 - true) * np.log(1 - predicted))
    
    @staticmethod
    def binary_cross_entropy_deriv(predicted, true):
        return (predicted - true) / (predicted * (1 - predicted))

class Layer:
    # Initialize layer with given settings, adding dropout support
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
            dA *= self.D  # Apply dropout mask
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

class Model:
    def __init__(self, lambda_val=0.0):
        self.layers = []
        self.lambda_val = lambda_val  # L2 Regularization parameter

    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def compute_loss(self, predicted, true):
        loss = LossFunction.binary_cross_entropy(predicted, true)
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
    
    def train(self, X_train, y_train, epochs, learning_rate, batch_size=64):
        lr = learning_rate
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            y_train_shuffled = y_train[:, permutation]

            for i in range(0, X_train.shape[1], batch_size):
                X_batch = X_train_shuffled[:, i:i+batch_size]
                Y_batch = y_train_shuffled[:, i:i+batch_size]
                
                predicted = self.forward(X_batch)
                
                loss = self.compute_loss(predicted, Y_batch)
                self.backward(predicted, Y_batch)
                
                self.update_params(lr)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

            if (epoch+1) % 500 == 0:
                lr /= 2
                print(f"Reduced learning rate to {lr}")

# Initialize model with adjusted architecture and dropout
model = Model(lambda_val=0.01)
model.add(Layer(n_input=784, n_neurons=64, activation='relu', dropout_rate=0.2))
model.add(Layer(n_input=64, n_neurons=32, activation='relu', dropout_rate=0.2))
model.add(Layer(n_input=32, n_neurons=10, activation='relu'))
model.add(Layer(n_input=10, n_neurons=1, activation='sigmoid'))

# Function implementations (get_predictions, get_accuracy, etc.) remain unchanged.

# Ensure Y_train is a 2D row vector for compatibility

if Y_train.ndim == 1:
    Y_train = Y_train.reshape(1, -1)
    
    
# Define the evaluate_model function
def evaluate_model(X, Y, model):
    """
    Evaluates the performance of the model on the given dataset X and true labels Y.
    Prints out the accuracy, precision, recall, and F1 score of the model predictions.
    """
    A_last = model.forward(X, training=False)  # Ensure dropout is not applied during evaluation
    predictions = get_predictions(A_last)
    
    accuracy = get_accuracy(predictions, Y.reshape(1, -1))
    precision = precision_score(Y.reshape(-1), predictions.reshape(-1))
    recall = recall_score(Y.reshape(-1), predictions.reshape(-1))
    f1 = f1_score(Y.reshape(-1), predictions.reshape(-1))
    
    print(f"Accuracy: {accuracy}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    
def get_predictions(output):
    return (output > 0.5).astype(int)

def get_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

if Y_train.ndim == 1:
    Y_train = Y_train.reshape(1, -1)

# Train the model with the new setup
model.train(X_train.T, Y_train, epochs=2000, learning_rate=0.005)

# Evaluate the model on the development set
evaluate_model(X_dev.T, Y_dev.reshape(1, -1), model)

# Output development set accuracy
output_dev = model.forward(X_dev.T, training=False)
predictions_dev = get_predictions(output_dev)
accuracy_dev = get_accuracy(predictions_dev, Y_dev.reshape(1, -1))
print(f"Development Set Accuracy: {accuracy_dev}%")

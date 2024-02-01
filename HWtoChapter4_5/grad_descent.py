from HWtoChapter4_5.activation import Activation


class GradDescent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_weights(self, layer, layer_gradients):
        for neuron, gradients in zip(layer.neurons, layer_gradients):
            neuron.weights -= self.learning_rate * gradients[1]
            neuron.bias -= self.learning_rate * gradients[0]

class Training:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train(self, inputs, targets):
        for x, y in zip(inputs, targets):
            # Forward propagation
            predictions = self.model.predict(x)

            # Compute loss and its derivative
            loss = self.loss_function(predictions, y)
            loss_derivative = self.loss_function.derivative(predictions, y)

            # Backward propagation
            error, hidden_gradients = self.model.hidden_layer.backpropagate(loss_derivative, Activation.sigmoid_derivative)
            _, output_gradients = self.model.output_layer.backpropagate(error, Activation.sigmoid_derivative)

            # Update weights using gradient descent
            self.optimizer.update_weights(self.model.hidden_layer, hidden_gradients)
            self.optimizer.update_weights(self.model.output_layer, output_gradients)

            # Optionally print loss here

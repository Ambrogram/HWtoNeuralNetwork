from HWtoChapter4_5.activation import Activation
from HWtoChapter4_5.layer import Layer


class Model:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize one hidden layer and one output layer
        self.hidden_layer = Layer(hidden_size, input_size)
        self.output_layer = Layer(output_size, hidden_size)

    def predict(self, inputs):
        # Forward propagation through the hidden layer
        hidden_output = self.hidden_layer.forward(inputs, Activation.sigmoid)

        # Forward propagation through the output layer
        final_output = self.output_layer.forward(hidden_output, Activation.sigmoid)
        return final_output

    # Additional methods for backpropagation and updating weights will go here

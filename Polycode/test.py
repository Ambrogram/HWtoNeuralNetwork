import numpy as np 

class Parameter: 
    def __init__(self):
        self.weights = None
        self.bias = None
        
        
    def set_bias(self, bias):
        self.bias = bias
        
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    
class Neuron:
    def __init__(self, input_size):
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self.agg_signal = None
        self.activation = None
        self.output = None
        
    def neuron(self):
        self.weights = np.random.randn(self.input_size)
        self.bias = np.random.randn()
        
    
        
        
class Activation:
    def __init__(self, neurons, parameters, activation_type):
        self.neurons = neurons
        self.parameters = parameters
        self.activation_type = activation_type
        self.output = None
        
        
        
        
class Layer:    
    def __init__(self, input_size, num_neuron, activation_type):
        self.input_size = input_size
        self.num_neuron = num_neuron
        self.weights = np.random.randn(input_size, num_neuron)
        self.neurons = [Neuron(input_size) for i in range(num_neuron)]
        self.parameters = Parameter()
        self.activation = Activation(self.neurons, self.parameters, activation_type)
        self.output = None
        
    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            neuron.weights = np.random.randn(self.neuron_layer)
            neuron.agg_signal = np.dot(inputs, neuron.weights) + neuron.bias
            neuron.activation = self.activation.activation_type(neuron.agg_signal)
            outputs.append(neuron.activation)
        
    def layer(self):
        self.parameters.set_bias(np.random.randn(self.num_neuron))
    

    
input_size = 2
num_neuron = 3

parameters = Parameter()
parameters.set_bias(np.random.randn(num_neuron))

inputs = np.random.randn(input_size, num_neuron)
print("Inputs: ", inputs)


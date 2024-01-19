# Define a class named 'Neuron' to simulate the behavior of a McCulloch and Pitts neuron
class Neuron:
    
    # The initialization method is called when a new instance of Neuron is created
    def __init__(self, weights, threshold):
        # 'self.weights' stores the weights for the neuron; these are provided when the neuron is instantiated
        self.weights = weights

        # 'self.threshold' stores the threshold value for the neuron; the neuron activates when its weighted input exceeds this threshold
        self.threshold = threshold

    # 'calculate_output' is a method to compute the output of the neuron based on given inputs
    def calculate_output(self, inputs):
        # 'weighted_sum' calculates the sum of the product of each input value and its corresponding weight
        # 'zip(self.weights, inputs)' pairs each weight with its corresponding input
        # 'sum(w * i for w, i in ...)' computes the sum of these products
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))

        # The method returns 1 if the 'weighted_sum' is greater than or equal to the 'threshold', otherwise returns 0
        # This is a basic step function used as an activation function in the neuron
        return 1 if weighted_sum >= self.threshold else 0

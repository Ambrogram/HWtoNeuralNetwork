# Import the 'unittest' module, Python's built-in library for unit testing
# 'unittest' provides a framework for writing and running tests
import unittest

# Import the 'Neuron' class from the neuron_model module to test its functionality
# This assumes that there is a file named 'neuron_model.py' containing a class 'Neuron'
from neuron_model import Neuron

# Define a class 'TestNeuron' that inherits from 'unittest.TestCase'
# 'unittest.TestCase' is a base class with methods to assert various conditions for testing
# 'TestNeuron' will contain all the test methods for testing the Neuron class
class TestNeuron(unittest.TestCase):

    # Define a method to test the output of the Neuron class
    # This method will contain specific test cases to validate the functionality of Neuron
    def test_neuron_output(self):
        # Define a list of weights to be used for creating a Neuron instance
        # These weights are parameters for the neuron's calculation
        weights = [0.5, -0.2, 0.8]

        # Define a threshold value for the neuron activation
        # The neuron activates (outputs 1) if its weighted input is equal to or above this threshold
        threshold = 0.5

        # Create an instance of the Neuron class with the specified weights and threshold
        # This neuron instance will be used for testing
        neuron = Neuron(weights, threshold)

        # Test case for input [1, 0, 0]
        # 'assertEqual' checks if the actual output of the neuron matches the expected output (1 in this case)
        # The expected output should be 1, as the weighted sum equals the threshold
        self.assertEqual(neuron.calculate_output([1, 0, 0]), 1, "Output should be 1 for input [1, 0, 0]")

        # Test case for input [1, 1, 1]
        # Similar to the previous test, but with different inputs
        # The expected output should be 1, as the weighted sum (1.1) is greater than the threshold (0.5)
        self.assertEqual(neuron.calculate_output([1, 1, 1]), 1, "Output should be 1 for input [1, 1, 1]")

# This conditional statement checks if the script is being run as the main program
# and not being imported as a module in another script
if __name__ == '__main__':
    # If the script is run directly (not imported), execute the test suite
    # 'unittest.main()' provides a command-line interface to the test script
    unittest.main()

class OutputHandler:
    def __init__(self, log_file='neuron_output.log'):
        # Initialize the OutputHandler with an optional log file name
        self.log_file = log_file

    def display_output(self, inputs, output):
        # Display the inputs and corresponding output in a readable format
        print(f"Inputs: {inputs} -> Neuron Output: {output}")

    def log_output(self, inputs, output):
        # Write the inputs and corresponding output to the specified log file
        with open(self.log_file, 'a') as file:
            file.write(f"Inputs: {inputs} -> Neuron Output: {output}\n")

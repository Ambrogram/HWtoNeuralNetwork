class Training:
    def __init__(self, model, loss_function, parameters):
        self.model = model
        self.loss_function = loss_function
        self.parameters = parameters

    def train(self, inputs, targets):
        for epoch in range(self.parameters.epochs):
            for x, y in zip(inputs, targets):
                # Forward Propagation
                predictions = self.model.predict(x)
                # Compute Loss
                loss = self.loss_function.mean_squared_error(predictions, y)
                # Back Propagation (to be implemented)
                # Update Weights (to be implemented using Gradient Descent)
                # Print epoch, loss, etc. for monitoring

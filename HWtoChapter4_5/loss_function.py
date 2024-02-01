import numpy as np


class LossFunction:
    @staticmethod
    def mean_squared_error(predictions, targets):
        return np.mean((predictions - targets) ** 2)

    @staticmethod
    def mean_squared_error_derivative(predictions, targets):
        return 2 * (predictions - targets) / targets.size

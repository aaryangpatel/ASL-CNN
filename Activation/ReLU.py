import numpy as np


class ReLU:
    """This class implements the Rectified Linear Unit (ReLU) activation function used throughout the network."""

    def __init__(self):
        self.input = None

    """ReLU activation function for forward propagation."""
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    """Back propagation of ReLU."""
    def backward(self, output_grad, learning_rate):
        # Generates new np array with 0s and 1s representing the result of the given condition
        relu_grad = np.where(self.input > 0, 1, 0)
        return output_grad * relu_grad

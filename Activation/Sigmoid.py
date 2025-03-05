import numpy as np


class Sigmoid:
    """This class implements the Sigmoid activation function used throughout the network."""

    def __init__(self):
        self.output = None

    """Sigmoid activation function for forward propagation."""
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    """Back propagation of Sigmoid."""
    def backward(self, input, learning_rate):
        return self.output * (1 - self.output)
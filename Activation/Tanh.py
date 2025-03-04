import numpy as np


class Tanh:
    """This class implements the Hyperbolic Tangent (Tanh) activation function used throughout the
    network."""

    """Tanh activation function for forward propagation."""
    def forward(self, input):
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))

    """Back propagation of Tanh."""
    def backward(self, input):
        return 1 - self.forward(self.forward(input)) ** 2

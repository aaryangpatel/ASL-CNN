import numpy as np


class Flatten:
    """This class implements the Flatten layer which converts a multidimensional array into a 1D array."""
    def __init__(self):
        self.input_shape = None

    """Flattens input into a 1D array."""
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()

    """Converts a flattened array back into the original input's shape."""
    def backward(self, output_grad, learning_rate):
        return output_grad.reshape(self.input_shape)

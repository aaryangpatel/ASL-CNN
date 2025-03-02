import numpy as np


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()

    def backward(self, input, learning_rate):
        return input.reshape(self.input_shape)

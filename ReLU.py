import numpy as np


class ReLU:
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, learning_rate):
        if input <= 0:
            return 0
        return 1

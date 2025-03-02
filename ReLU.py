import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_grad, learning_rate):
        grad_relu = np.where(self.input > 0, 1, 0)
        return output_grad * grad_relu

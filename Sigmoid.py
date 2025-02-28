import numpy as np


class Sigmoid:
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input):
        return self.forward(input) * (1 - self.forward(input))
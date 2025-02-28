import numpy as np


class Tanh:
    def forward(self, input):
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
    
    def backward(self, input):
        return 1 - self.forward(self.forward(input)) ** 2
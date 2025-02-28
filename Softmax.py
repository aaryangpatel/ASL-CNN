import numpy as np


class Softmax:
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        self.output = np.exp(input) / np.sum(np.exp(input))
        return self.output
    
    def backward(self, output_grad):
        pass

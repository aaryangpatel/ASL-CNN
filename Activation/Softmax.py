import numpy as np


class Softmax:
    """This class implements the softmax which maps the Fully Connected Layer's output to a probability distribution
    for each of the labels."""
    def __init__(self):
        self.output = None
    
    def forward(self, input):
        exps = np.exp(input - np.max(input))
        self.output = exps / np.sum(exps)
        return self.output

    def backward(self, input, learning_rate):
        jacob = np.diag(self.output) - np.outer(self.output, self.output)
        return np.dot(jacob, input)

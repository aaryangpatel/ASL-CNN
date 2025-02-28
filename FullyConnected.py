import numpy as np


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.rand(self.input_size, self.output_size)
        self.biases = np.random.rand(self.output_size)
        
        self.d_weights = None

        self.input = None
        self.output = None
        
    def forward(self, input):
        self.input = input
        self.output += np.dot(self.input, self.weights) + self.biases
        return self.output
           
    def backward(self, output_grad, learning_rate):
        weights_gradient = np.dot(output_grad, self.input.T)
        input_gradient = np.dot(self.weights.T, output_grad)
        
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_grad

        return input_gradient

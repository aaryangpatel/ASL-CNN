import numpy as np


class FullyConnected:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.rand(self.input_size, self.output_size)
        self.biases = np.zeros(self.output_size)

        self.input = None
        self.output = None
        
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, output_grad, learning_rate):
        weights_grad = np.dot(self.input.reshape(-1, 1), output_grad.reshape(1, -1))
        input_grad = np.dot(output_grad, self.weights.T).reshape(self.input.shape)

        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * output_grad

        return input_grad

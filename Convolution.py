import numpy as np
from scipy import signal


class Convolution:
    def __init__(self, input_shape, output_depth, kernel_size):
        self.input_shape = input_shape
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        input_depth, input_height, input_width = self.input_shape
        self.input_depth = input_depth

        self.output_shape = (output_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (output_depth, input_depth, kernel_size, kernel_size)

        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.zeros(self.output_shape)

        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for neuron in range(self.output_depth):
            for channel in range(self.input_depth):
                self.output[neuron] += signal.correlate2d(self.input[channel], self.kernels[neuron, channel], "valid")

        return self.output

    def backward(self, output_grad, learning_rate):
        kernel_grad = np.zeros_like(self.kernels)
        input_grad = np.zeros_like(self.input)

        for neuron in range(self.output_depth):
            for channel in range(self.input_depth):
                kernel_grad[neuron, channel] += signal.correlate2d(self.input[channel], output_grad[neuron],
                                                                   "valid")
                input_grad[channel] += signal.convolve2d(output_grad[neuron], self.kernels[neuron, channel], "full")

        self.kernels -= learning_rate * kernel_grad
        self.biases -= learning_rate * output_grad

        return input_grad

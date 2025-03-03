import numpy as np
from scipy import signal


class Convolution:
    """This class implements the convolutional stage of the CNN which emphasizes the image's central features."""
    def __init__(self, input_shape, output_depth, kernel_size):
        self.input_shape = input_shape
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        input_depth, input_height, input_width = self.input_shape
        self.input_depth = input_depth

        self.output_shape = (output_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernel_shape = (output_depth, input_depth, kernel_size, kernel_size)

        # Uses the Xavier method of initializing weights with a uniform distribution
        self.kernels = np.random.randn(*self.kernel_shape) * np.sqrt(1 / (input_depth * kernel_size * kernel_size))
        self.biases = np.zeros(self.output_shape)

        self.input = None
        self.output = None

    """Computes the convoluted matrix by overlaying the input image with the kernel."""
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)

        for output_channel in range(self.output_depth):
            for input_channel in range(self.input_depth):
                self.output[output_channel] += signal.correlate2d(self.input[input_channel], self.kernels[output_channel, input_channel], "valid")

        return self.output


    """Computes the gradients and adjusts kernels and biases and the input"""
    def backward(self, output_grad, learning_rate):
        kernel_grad = np.zeros_like(self.kernels)
        input_grad = np.zeros_like(self.input)

        for output_channel in range(self.output_depth):
            for input_channel in range(self.input_depth):
                kernel_grad[output_channel, input_channel] += signal.correlate2d(self.input[input_channel], output_grad[output_channel], "valid")
                input_grad[input_channel] += signal.convolve2d(output_grad[output_channel], self.kernels[output_channel, input_channel], "full")

        self.kernels -= learning_rate * kernel_grad
        self.biases -= learning_rate * output_grad

        return input_grad

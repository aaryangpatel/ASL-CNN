import numpy as np
import math


class MaxPooling:
    """This class implements the Max Pooling layer which reduces the image's dimensions."""

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None

    """Reduces the dimensions of the input image by selecting the maximum element in each pool window."""
    def forward(self, input):
        self.input = input
        input_depth, input_height, input_width = input.shape

        output_height = math.floor((input_height - self.pool_size) / self.stride) + 1
        output_width = math.floor((input_width - self.pool_size) / self.stride) + 1
        output = np.zeros((input_depth, output_height, output_width))

        for channel in range(input_depth):
            # Iterate over each possible overlap of the pool window which is shifted stride units each iteration
            for row in range(0, input_height - self.pool_size + 1, self.stride):
                for col in range(0, input_width - self.pool_size + 1, self.stride):
                    # Extracts the overlap of the pool window and the original input image
                    overlap = self.input[channel, row: row + self.pool_size, col: col + self.pool_size]
                    # Enter the maximum element into the output matrix with lower dimensions than the input image
                    output[channel, math.floor(row / self.stride), math.floor(col / self.stride)] = np.max(overlap)

        return output

    """Updates the gradients of the max elements selected in the Max Pooling forward propagation."""
    def backward(self, output_grad, learning_rate):
        input_grad = np.zeros_like(self.input)
        input_depth, input_height, input_width = self.input.shape

        # Iterates in the similar fashion to the forward propagation
        for channel in range(input_depth):
            for row in range(0, input_height - self.pool_size + 1, self.stride):
                for col in range(0, input_width - self.pool_size + 1, self.stride):
                    overlap = self.input[channel, row: row + self.pool_size, col: col + self.pool_size]
                    # Flattens to find the index of the max element and scales it to the index in the overlap region
                    max_index = np.unravel_index(np.argmax(overlap, axis=None), overlap.shape)
                    # Adds corresponding element of output grad to the input grad at the max element's index
                    input_grad[channel, row + max_index[0], col + max_index[1]] = output_grad[channel, math.floor(row / self.stride), math.floor(col / self.stride)]

        return input_grad

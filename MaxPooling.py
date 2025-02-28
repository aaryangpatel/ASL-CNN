import numpy as np
import math


class MaxPooling:
    def __init__(self, pool_size, stride, padding):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        self.input = None
        
    def forward(self, input):
        self.input = input
        
        input_depth, input_height, input_width = input.shape
        output_height = math.floor((input_height - self.pool_size) / self.stride)
        output_width = math.floor((input_width - self.pool_size) / self.stride)
        output = np.zeros(input_depth, output_height, output_width)
        
        for channel in range(input_depth):
            for row in range(input_height - self.pool_size + 1, self.stride):
                for col in range(input_width - self.pool_size + 1, self.stride):
                    overlap = self.input[channel, row : row + self.pool_size, col : col + self.pool_size]
                    output[channel, math.floor(row / self.stride), math.floor(col / self.stride)] = np.max(overlap)
                    
        return output      
        
    def backward(self, output_grad, learning_rate):
        input_grad = np.zeros_like(self.input)
        input_depth, input_height, input_width = self.input.shape

        for channel in range(input_depth):
            for row in range(input_height - self.pool_size + 1, self.stride):
                for col in range(input_width - self.pool_size + 1, self.stride):
                    overlap = self.input[channel, row : row + self.pool_size, col : col + self.pool_size]
                    # Flattens to find the max element and then transforms back to matrix
                    max_index = np.unravel_index(np.argmax(overlap, axis=None), overlap.shape)
                    input_grad[channel, row + max_index[0], col + max_index[1]] = output_grad[channel, math.floor(row / self.stride), math.floor(col / self.stride)]
                    
        return input_grad

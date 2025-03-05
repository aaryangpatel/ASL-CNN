import numpy as np


class FullyConnected:
    """This class implements the final major part of a CNN, the Fully Connected Layer or Dense Layer."""

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Uses the Xavier method of initializing weights with a uniform distribution
        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / input_size)
        self.biases = np.zeros(self.output_size)

        self.input = None
        self.output = None

    """Computes the linear combination of the input with the layer's weights and biases."""
    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    """Computes the gradients for the weights, biases, and input to update the layer's properties and provide the 
    input for the next layer in back propagation."""
    def backward(self, output_grad, learning_rate):
        # Weights gradient from the dot product of flattened input and output gradient
        weights_grad = np.dot(self.input.reshape(-1, 1), output_grad.reshape(1, -1))
        # Input gradient from the dot product of the output gradient and transposed weights reshaped into the layer's input shape
        input_grad = np.dot(output_grad, self.weights.T).reshape(self.input.shape)

        # Update the weights and biases using the gradients and learning rate
        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * output_grad

        return input_grad

import numpy as np


class Network:
    """This class implements the core of the neural network including the functionality to train and predict on given
    data."""
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss

    """Predicts the label for a given input using forward propagation on each of the network's layers."""
    def predict(self, input):
        # Forward propagate starting with input
        output = input

        for layer in self.network:
            output = layer.forward(output)

        return output

    """Trains the model by updating the weights and biases of each layer through back propagation."""
    def train(self, x_train, y_train, epochs, learning_rate):
        # Iterate through each epoch
        for epoch in range(epochs):
            print("On epoch", epoch)
            # Iterate through each training sample
            for i in range(x_train.shape[0]):
                output = self.predict(x_train[i])

                # Back propagate starting with error gradient
                grad = self.loss.backward(y_train[i], output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
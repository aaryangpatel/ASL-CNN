import numpy as np


class Network:
    def __init__(self, network, loss):
        self.network = network
        self.loss = loss

    def predict(self, input):
        output = input
        for layer in self.network:
            output = layer.forward(output)

        return output

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            error = 0
            for i in range(x_train.shape[0]):
                output = self.predict(x_train[i])

                error += self.loss.forward(y_train[i], output)

                grad = self.loss.backward(y_train[i], output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
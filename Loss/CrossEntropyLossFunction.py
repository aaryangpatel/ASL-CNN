import numpy as np


class CrossEntropyLossFunction:
    """This class implements the cross entropy loss function which is used for training and evaluating the model."""

    def forward(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-10))

    def backward(self, y_true, y_pred):
        return y_pred - y_true

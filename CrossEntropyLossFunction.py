import numpy as np


class CrossEntropyLossFunction:
    def forward(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))
    
    def backward(self, y_true, y_pred):
        return y_pred - y_true

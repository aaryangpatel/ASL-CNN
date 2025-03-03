import numpy as np


class MSELossFunction:
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def d_mse(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

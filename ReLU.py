import numpy as np


class ReLU:
    def forward(self, input):
        return max(0, input)

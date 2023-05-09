import numpy as np

class ReLU():
    def forward_pass(self, x):
        return np.where(x >= 0, x, 0)
import numpy as np

class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        sig = Sigmoid.forward(x)
        return sig * (1 - sig)
class Tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x):
        return 1 - np.tanh(x) ** 2
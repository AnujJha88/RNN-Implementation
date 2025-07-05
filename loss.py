import numpy as np

class CrossEntropyLoss:
    def forward(self, y, y_pred):
        return -np.sum(y * np.log(y_pred))
    def backward(self, y, y_pred):
        return y_pred - y

class MSELoss:
    def forward(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)
    def backward(self, y, y_pred):
        return 2 * (y_pred - y) / y.shape[0]

class BCELoss:
    def forward(self, y, y_pred):
        return -np.sum(y * np.log(y_pred))
    def backward(self, y, y_pred):
        return y_pred - y
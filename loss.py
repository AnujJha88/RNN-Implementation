import numpy as np

class CrossEntropyLoss:
    def forward(self, y, y_pred):
        epsilon = 1e-9
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Standard Cross Entropy Formula
        return -np.sum(y * np.log(y_pred)) / y.shape[0]

    def backward(self, y, y_pred):
        # Gradient assumes Softmax activation in the previous layer
        return (y_pred - y) / y.shape[0]

class MSELoss:
    def forward(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def backward(self, y, y_pred):
        return 2 * (y_pred - y) / y.shape[0]

class BCELoss:
    def forward(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Binary Cross Entropy requires both terms: y and (1-y)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1-y_pred))
        return loss

    def backward(self, y, y_pred):
        # Gradient assumes Sigmoid activation in the previous layer
        return (y_pred - y) / y.shape[0]
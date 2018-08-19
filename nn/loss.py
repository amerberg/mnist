import numpy as np
from .tools import softmax


class Loss:
    def __init__(self):
        self.y_true = None

    def set_true_value(self, y_true):
        self.y_true = y_true

    def gradient(self, y_pred):
        raise NotImplementedError()

    def compute(self, y_pred):
        raise NotImplementedError()


class SoftMaxCrossEntropyWithLogits(Loss):
    def gradient(self, y_pred):
        return (softmax(y_pred) - self.y_true) / y_pred.shape[0]

    def compute(self, y_pred, y_true=None):
        if y_true is None:
            y_true = self.y_true
        return np.mean(np.sum(y_true * np.log(softmax(y_pred)),  axis=1))

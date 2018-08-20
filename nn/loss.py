import numpy as np
from .tools import softmax
from abc import abstractmethod


class Loss(object):
    def __init__(self):
        self.y_true = None

    def set_true_value(self, y_true):
        self.y_true = y_true

    @abstractmethod
    def gradient(self, y_pred):
        pass

    @abstractmethod
    def compute(self, y_pred):
        pass


class SoftMaxCrossEntropyWithLogits(Loss):
    """ Softmax cross entropy with logits. Do not use a softmax activation function
        on the last layer of a network that uses this loss function."""
    def gradient(self, y_pred):
        return (softmax(y_pred) - self.y_true) / y_pred.shape[0]

    def compute(self, y_pred, y_true=None):
        if y_true is None:
            y_true = self.y_true
        return - np.mean(np.sum(y_true * np.log(softmax(y_pred)), axis=1))

import numpy as np

class Activation(object):
    """ Activation function base class"""

    @staticmethod
    def compute(x):
        raise NotImplementedError()

    @staticmethod
    def derivative(x):
        raise NotImplementedError()


class ReLU(Activation):
    @staticmethod
    def compute(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x):
        return (x > 0).astype(np.float32)


class Identity(Activation):
    """ This activation function does nothing. Use it when you want a layer with no activation."""

    @staticmethod
    def compute(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(x.shape)

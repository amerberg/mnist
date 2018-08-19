import numpy as np


class Optimizer(object):
    """ Optimizer base class"""

    def update_layer(self, layer):
        """ Override this to update the parameters of a single layer."""
        raise NotImplementedError()


class AdaGrad(Optimizer):
    """ Implements AdaGrad as documented at http://ruder.io/optimizing-gradient-descent/"""

    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def update_layer(self, layer):
        """ Update a single layer, caching the necessary data in the layer."""
        g = layer.get_persistent_cache('adagrad_g')
        params = layer.parameters()
        grads = layer.gradients()
        if g is None:
            g = {name: np.zeros(val.shape) for name, val in params.items()}

        for key, param in params.items():
            # Update the cached value g
            g[key] += grads[key] ** 2
            # Update the parameter values
            params[key] -= self.learning_rate * grads[key] / (g[key] + self.epsilon) ** 0.5

        layer.set_persistent_cache('adagrad_g', g)


class GradientDescent(Optimizer):
    """ Simple implementation of batch gradient descent."""

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_layer(self, layer):
        params = layer.parameters()
        grads = layer.gradients()

        for key, param in params.items():
            params[key] -= self.learning_rate * grads[key]

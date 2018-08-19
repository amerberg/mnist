import numpy as np

class Optimizer(object):
    def train_step(self, model, X, Y):
        model.feedforward(X, Y)
        model.backpropagate()
        for layer in model.layers:
            self.update_parameters(layer)


class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.001, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def update_layer(self, layer):
        g = layer.get_persistent_cache('adagrad_g')
        params = layer.parameters()
        grads = layer.gradients()
        if g is None:
            g = {name: np.zeros(val.shape) for name, val in params.items()}

        for key, param in params.items():
            g[key] += grads[key] ** 2
            params[key] -= self.learning_rate * grads[key] / (g[key] + self.epsilon) ** 0.5

        layer.set_persistent_cache('adagrad_g', g)


class GradientDescent(Optimizer):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_layer(self, layer):
        params = layer.parameters()
        grads = layer.gradients()

        for key, param in params.items():
            params[key] -= self.learning_rate * grads[key]


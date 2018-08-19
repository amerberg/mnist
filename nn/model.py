import numpy as np


class Model(object):
    """Model class which handles interactions between layers, training, optimization, etc."""
    def __init__(self, layers, loss, optimizer):
        for layer in layers:
            layer.set_model(self)

        for previous_layer, layer in zip(layers, layers[1:]):
            layer.set_previous(previous_layer)
            previous_layer.set_next(layer)

        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.batch_number = 0

    def fit(self, X, Y, n_batches=10, batch_size=100):
        assert X.shape[1:] == self.layers[0].input_shape, "Wrong input shape"
        assert X.shape[0] == Y.shape[0], "Non-matching shapes"

        for batch_number in range(n_batches):
            batch_ind = np.random.randint(X.shape[0], size=batch_size)
            batch_X = X[batch_ind, ...]
            batch_Y = Y[batch_ind]
            self.fit_batch(batch_X, batch_Y)

    def fit_batch(self, X, Y):
        self.batch_number += 1
        self.forward(X, training=True)
        self.backward(Y)

    def predict(self, X):
        return self.forward(X, training=False)

    def forward(self, X, training=False):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, Y):
        self.loss.set_true_value(Y)
        for layer in reversed(self.layers):
            self.optimizer.update_layer(layer)

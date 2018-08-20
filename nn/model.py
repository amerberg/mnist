import numpy as np
from .layer import Dense
from .tools import NotYetSupportedError

class Model(object):
    """Model class which handles interactions between layers, training, optimization, etc."""

    def __init__(self, layers, loss, optimizer):
        """ layers: a list of nn.layer.Layer objects
            loss: a nn.loss.Loss object
            optimizer: an nn.optimizer.Optimizer object """
        for layer in layers:
            layer.set_model(self)

        for previous_layer, layer in zip(layers, layers[1:]):
            layer.set_previous(previous_layer)
            previous_layer.set_next(layer)

        if not isinstance(layers[-1], Dense):
            raise NotYetSupportedError("The last layer must be fully connected.")

        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.batch_number = 0

    def fit(self, X, Y, n_batches=10, batch_size=100):
        """ Fit a model to the specified data, with the specified number and size of batches."""
        assert X.shape[1:] == self.layers[0].input_shape, "Wrong input shape"
        assert X.shape[0] == Y.shape[0], "Non-matching shapes"

        for batch_number in range(n_batches):
            batch_ind = np.random.randint(X.shape[0], size=batch_size)
            batch_X = X[batch_ind, ...]
            batch_Y = Y[batch_ind]
            self.fit_batch(batch_X, batch_Y)

    def fit_batch(self, X, Y):
        """ Train the model on a single batch of data."""
        self.batch_number += 1
        self.forward(X, training=True)
        self.backward(Y)

    def predict(self, X):
        """ Generate predictions for the given data."""
        return self.forward(X, training=False)

    def forward(self, X, training=False):
        """ A forward pass of the data. Can either be training or prediction."""
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, Y):
        """ A backward pass of the data. Updates all parameters with the model's optimizer."""
        self.loss.set_true_value(Y)
        for layer in reversed(self.layers):
            self.optimizer.update_layer(layer)

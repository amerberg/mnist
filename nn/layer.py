import numpy as np
from functools import reduce
from operator import mul
from .tools import filter2d, zero_pad, NotYetSupportedError
from .activation import Identity
from abc import abstractmethod


def cached(key):
    """ A decorator for temporarily caching function values during backpropagation."""

    def decorator(fn):
        def decorated(cls):
            value = cls.get_cache(key)
            if value is not None:
                return value
            else:
                value = fn(cls)
                cls.set_cache(key, value)
                return value

        return decorated

    return decorator


class Layer(object):
    """ A base class for layer objects. This implements some basic functionality for
        interacting with adjacent layers and for caching values."""

    def __init__(self):
        self._cache = {}
        self._persistent_cache = {}
        self.next_layer = None
        self.previous_layer = None
        self.input_shape = None
        self.output_shape = None
        self.model = None

    def parameters(self):
        """ Override this method to return the current parameter values for the layer. """
        return {}

    def gradients(self):
        """ Override this method to return the gradient of the loss function with respect to
            the layer's parameters. """
        return {}

    def set_model(self, model):
        """ Keep track of the model."""
        self.model = model

    def set_previous(self, previous_layer):
        """ Set the previous layer. Override this if you need to do anything beyond setting
            input shape, e.g. to initialize parameters based on the input shape."""
        self.previous_layer = previous_layer
        self.input_shape = previous_layer.output_shape

    def set_next(self, next_layer):
        """ Keep track of the next layer in the model. Override this if you need to initialize
            anything when the next layer is set."""
        self.next_layer = next_layer

    def set_cache(self, key, value):
        """ Store a value in the short-term cache, which is not saved between batches."""
        self._cache[key] = (self.model.batch_number, value)

    def get_cache(self, key, default=None):
        """ Get a value from the short-term cache. """
        batch_number, value = self._cache.get(key, (None, None))
        if batch_number == self.model.batch_number:
            return value
        else:
            return default

    def set_persistent_cache(self, key, value):
        """ Store a value in a cache which persists between batches."""
        self._persistent_cache[key] = value

    def get_persistent_cache(self, key, default=None):
        """ Get a value from the persistent cache."""
        return self._persistent_cache.get(key, default)

    @abstractmethod
    def forward(self, X, training=False):
        """ Complete a forward pass of the net and return the output. Set training=true
            to cache data needed for backpropagation. Must be overridden for each layer class."""
        pass

    @abstractmethod
    def d_input(self):
        """ Use cached data to compute gradients to send to the previous layer during backpropagation.
            This will be specific to each layer and must be overridden."""
        pass


class Conv2D(Layer):
    """ A 2-d convolutional layer."""
    def __init__(self, filter_size, stride=(1, 1), input_shape=None, channels=1, activation=Identity(), padding='valid',
                 filter_initializer=None, bias_initializer=None):
        if stride != (1, 1):
            raise NotYetSupportedError('Only a stride of (1, 1) is currently supported.')

        if not (padding == 'same'):
            raise NotYetSupportedError('Only same padding is currently supported for convolutional layers.')

        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.channels = channels
        self.activation = activation
        self.padding = padding
        self.input_shape = input_shape
        self.filter_initializer = filter_initializer
        self.bias_initializer = bias_initializer
        self.filter = None
        self.bias = bias_initializer((self.channels,))

        if input_shape is not None:
            self.set_output_shape()
            self.initialize_filter()
        else:
            # If input_shape is not provided, it will be computed when the previous layer is set.
            self.output_shape = None
            self.filter = None

    def forward(self, X, training=False):
        """ Forward pass using the standard convolution formulas."""
        weighted_input = filter2d(zero_pad(X, self.filter.shape[0:2], self.padding), self.filter,
                                  self.stride) + self.bias
        activation = self.activation.compute(weighted_input)
        if training:
            self.set_cache('weighted_input', weighted_input)
            self.set_cache('activation', activation)
            self.set_cache('input', X)
        return activation

    def set_previous(self, previous_layer):
        """ Set the output shape and initialize the filter."""
        super().set_previous(previous_layer)
        self.set_output_shape()
        self.initialize_filter()

    def initialize_filter(self):
        """ Compute the filter shape and call the initializer function."""
        shape = self.filter_size + (self.input_shape[-1], self.channels)
        self.filter = self.filter_initializer(shape)

    def set_output_shape(self):
        input_shape = self.input_shape
        channels = self.channels
        padding = self.padding
        filter_size = self.filter_size
        stride = self.stride
        if padding == 'same':
            self.output_shape = input_shape[0], input_shape[1], channels
        elif padding == 'valid':
            self.output_shape = ((input_shape[0] - filter_size[0] + 1) // stride[0],
                                 (input_shape[1] - filter_size[1] + 1) // stride[0],
                                 channels)
        # TODO: full padding

    @cached('error')
    def error(self):
        """ Don't """
        d_a = self.next_layer.d_input()
        weighted_input = self.get_cache('weighted_input')
        return d_a * self.activation.derivative(weighted_input)

    def d_input(self):
        """ Cross-correlate the error with the rotated, transposed filter."""
        padded_error = zero_pad(self.error(), self.filter.shape[:2], self.padding)
        return filter2d(padded_error, np.rot90(self.filter.transpose(0, 1, 3, 2), 2), self.stride)

    def gradients(self):
        return {
            'filter': self.filter_gradient(),
            'bias': np.sum(self.error(), axis=(0, 1))
        }

    def filter_gradient(self):
        """ Compute the gradient of the loss with respect to the gradient. Inelegant but it seems to work."""
        error = self.error()
        error_shape = error.shape[1:3]
        input_ = self.get_cache('input')
        h = input_.shape[1]
        w = input_.shape[2]
        # TODO: check that this padding is correct. I am not sure it gets it right when the number of 0s to add is odd.
        pad_l, pad_t = (error_shape[0] - 1) // 2, (error_shape[1] - 1) // 2
        pad_r, pad_b = (error_shape[0] - 1) - pad_l, (error_shape[1] - 1) - pad_t
        padded_input = np.pad(input_, ((0, 0), (pad_l, pad_r), (pad_t, pad_b), (0, 0)), 'constant')
        # TODO: this assumes 'same' padding. support other types.
        # TODO: I wanted to use np.from_function but kept getting an IndexError, so we'll use a nested for loop for now
        grads = np.zeros(self.filter.shape)
        for m in range(grads.shape[0]):
            for n in range(grads.shape[1]):
                for k in range(grads.shape[2]):
                    for r in range(grads.shape[3]):
                        grads[m, n, k, r] = np.sum(error[:, :, :, r] * padded_input[:, m: m + h, n: n + w, k])
        return grads


class MaxPool2D(Layer):
    """A max pooling layer."""

    def __init__(self, pool_size, stride=None, padding='valid', input_shape=None):
        super().__init__()
        self.pool_size = pool_size
        self.padding = padding  # TODO: padding is not yet implemented.
        self.input_shape = input_shape
        if input_shape is not None:
            self.set_output_shape()
        else:
            self.output_shape = None
        if stride is not None:
            self.stride = stride
        else:
            self.stride = pool_size

        if self.stride != self.pool_size:
            raise NotYetSupportedError("Pool size and stride must be the same.")

    def set_previous(self, previous_layer):
        """ Set the output shape and check that we haven't been given unsupported values."""
        super().set_previous(previous_layer)
        self.set_output_shape()

        if self.input_shape[0] % self.pool_size[0] != 0 or self.input_shape[1] % self.pool_size[1] != 0:
            raise NotYetSupportedError('Input shape must be an integer multiple of pool size.')

    def set_output_shape(self):
        """ Determine output shape based on input shape and stride. """
        self.output_shape = ((self.input_shape[0] // self.stride[0],
                              self.input_shape[1] // self.stride[1],
                              self.input_shape[2]
                              ))

    def forward(self, X, training=False):
        """"""
        output = np.zeros((X.shape[0],) + self.output_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                output[:, i, j, :] = np.max(X[:, i * self.stride[0]:i * self.stride[0] + self.pool_size[0],
                                            j * self.stride[1]:j * self.stride[1] + self.pool_size[1], :], axis=(1, 2))
        if training:
            self.set_cache('output', output)
            self.set_cache('input', X)
        return output

    @cached('error')
    def error(self):
        return self.next_layer.d_input()

    def d_input(self):
        """ Compute the gradient with respect to the inputs (which are also the activations of the
            preceding layer)."""
        input_ = self.get_cache('input')
        output = self.get_cache('output')
        error = self.error()
        result = np.zeros((error.shape[:1] + self.input_shape))
        stride_x, stride_y = self.stride
        for i in range(self.input_shape[0]):
            for j in range(self.input_shape[1]):
                result[:, i, j, :] = (output[:, i // stride_x, j // stride_y, :] == input_[:, i, j, :]
                                      ) * error[:, i // stride_x, j // stride_y, :]
        return result


class Dense(Layer):
    """ This class implements a fully-connected layer."""

    def __init__(self, size, weight_initializer, bias_initializer, input_shape=None, activation=Identity()):
        super().__init__()
        self.size = size
        self.activation = activation
        self.input_shape = input_shape
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.output_shape = (size,)
        self.weight = None
        self.bias = self.bias_initializer(self.output_shape)
        if input_shape is not None:
            self.weight = self.weight_initializer((self.input_shape[0], self.output_shape[0]))

    def set_previous(self, previous_layer):
        """ Initialize the weight. """
        super().set_previous(previous_layer)
        self.weight = self.weight_initializer((self.input_shape[0], self.output_shape[0]))

    @cached('error')
    def error(self):
        if self.next_layer is not None:
            # This is not the last layer. Get d_input from the next one.
            d_a = self.next_layer.d_input()
            weighted_input = self.get_cache('weighted_input')
            return d_a * self.activation.derivative(weighted_input)
        else:
            # This is the last layer, so get the gradient of the cost function.
            grad = self.model.loss.gradient(self.get_cache('activation'))
            return grad * self.activation.derivative(self.get_cache('weighted_input'))

    def d_input(self):
        return np.matmul(self.error(), self.weight.T)

    def gradients(self):
        """ Compute the gradients based on the standard formulas."""
        error = self.error()
        input_ = self.get_cache('input')
        error_stacked = np.reshape(error, (error.shape[0], 1, error.shape[1]))
        input_stacked = np.reshape(input_, (input_.shape[0], input_.shape[1], 1))
        return {
            'bias': np.sum(error, axis=0),
            'weight': np.sum(np.matmul(input_stacked, error_stacked), axis=0)
        }

    def parameters(self):
        return {
            'bias': self.bias,
            'weight': self.weight
        }

    def forward(self, X, training=False):
        """ Basic forward pass for a fully connected layer."""
        weighted = np.matmul(X, self.weight) + self.bias
        activation = self.activation.compute(weighted)
        if training:
            self.set_cache('weighted_input', weighted)
            self.set_cache('activation', activation)
            self.set_cache('input', X)
        return activation


class Flatten(Layer):
    """ A layer which flattens its input."""
    def forward(self, inputs, training=False):
        return np.reshape(inputs, (inputs.shape[0], -1))

    def set_previous(self, previous_layer):
        super().set_previous(previous_layer)
        self.set_output_shape()

    def set_output_shape(self):
        """ The output shape is just the product of all of the input dimensions."""
        self.output_shape = (reduce(mul, self.input_shape),)

    def d_input(self):
        """ Just convert the next layer's d_input back to the input shape."""
        return np.reshape(self.next_layer.d_input(), (-1,) + self.input_shape)

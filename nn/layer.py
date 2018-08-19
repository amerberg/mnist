import numpy as np
from functools import reduce
from operator import mul
from .activation import Identity


class NotYetSupportedError(Exception):
    pass


def filter2d(input_, filter_, stride=(1, 1)):
    h_strides = (input_.shape[1] - filter_.shape[0]) // stride[0] + 1
    v_strides = (input_.shape[2] - filter_.shape[1]) // stride[1] + 1
    result = np.zeros((input_.shape[0], h_strides, v_strides, filter_.shape[-1]))
    for i in range(h_strides):
        for j in range(v_strides):
            result[:, i, j, :] = np.sum(input_[:, i * stride[0]:i * stride[0] + filter_.shape[0],
                                        j * stride[1]:j * stride[1] + filter_.shape[1], :,
                                        np.newaxis] * filter_, axis=(1, 2, 3))

    return result


def cached(key):
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
    def __init__(self):
        self._cache = {}
        self._persistent_cache = {}
        self.next_layer = None
        self.previous_layer = None
        self.input_shape = None
        self.output_shape = None
        self.model = None

    def parameters(self):
        return {}

    def gradients(self):
        return {}

    def set_model(self, model):
        self.model = model

    def set_previous(self, previous_layer):
        self.previous_layer = previous_layer
        self.input_shape = previous_layer.output_shape

    def set_next(self, next_layer):
        self.next_layer = next_layer

    def set_cache(self, key, value):
        self._cache[key] = (self.model.batch_number, value)

    def get_cache(self, key, default=None):
        batch_number, value = self._cache.get(key, (None, None))
        if batch_number == self.model.batch_number:
            return value
        else:
            return default

    def set_persistent_cache(self, key, value):
        self._persistent_cache[key] = value

    def get_persistent_cache(self, key, default=None):
        return self._persistent_cache.get(key, default)


class Conv2D(Layer):
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
        self.bias = bias_initializer((self.channels, ))

        if input_shape is not None:
            self.set_output_shape()
            self.initialize_filter()
        else:
            self.output_shape = None
            self.filter = None

    def forward(self, X, training=True):
        weighted_input = filter2d(self.pad(X), self.filter, self.stride) + self.bias
        activation = self.activation.compute(weighted_input)
        if training:
            self.set_cache('weighted_input', weighted_input)
            self.set_cache('activation', activation)
        return activation

    def set_previous(self, previous_layer):
        super().set_previous(previous_layer)
        self.set_output_shape()
        self.initialize_filter()

    def initialize_filter(self):
        shape = self.filter_size + (self.input_shape[-1], self.channels)
        self.filter = self.filter_initializer(shape)

    def pad(self, input_):
        if self.padding == 'valid':
            return input_
        elif self.padding == 'same':
            l_pad, t_pad = (self.filter_size[0] - 1) // 2, (self.filter_size[1] - 1) // 2
            r_pad, b_pad = self.filter_size[0] - 1 - l_pad, self.filter_size[1] - 1 - t_pad
            return np.pad(input_, ((0, 0), (l_pad, r_pad), (t_pad, b_pad), (0, 0)), 'constant')
        elif self.padding == 'full':
            h_pad, v_pad = self.filter_size[0] - 1, self.filter_size[1] - 1
            return np.pad(input_, ((0, 0), (h_pad, h_pad), (v_pad, v_pad), (0, 0)), 'constant')

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

    @cached('error')
    def error(self):
        d_a = self.next_layer.d_input()
        weighted_input = self.get_cache('weighted_input')
        return d_a * self.activation.derivative(weighted_input)

    def d_input(self):
        return filter2d(self.pad(self.error()), np.rot90(self.filter.transpose(0, 1, 3, 2), 2), self.stride)

    def gradients(self):
        return {
            'filter': np.sum(filter2d(self.error(), self.filter.transpose(0, 1, 3, 2), stride=self.stride), axis=0),
            'bias': np.sum(self.error(), axis=(0, 1))
        }


class MaxPool2D(Layer):
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
        super().set_previous(previous_layer)
        self.set_output_shape()

        if self.input_shape[0] % self.pool_size[0] != 0 or self.input_shape[1] % self.pool_size[1] != 0:
            raise NotYetSupportedError('Input must be an integer multiple of pool size.')

    def set_output_shape(self):
        self.output_shape = ((self.input_shape[0] // self.stride[0],
                 self.input_shape[1] // self.stride[1],
                 self.input_shape[2]
                 ))

    def forward(self, X, training=True):
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
        super().set_previous(previous_layer)
        self.weight = self.weight_initializer((self.input_shape[0], self.output_shape[0]))

    @cached('error')
    def error(self):
        if self.next_layer is not None:
            d_a = self.next_layer.d_input()
            weighted_input = self.get_cache('weighted_input')
            return d_a * self.activation.derivative(weighted_input)
        else:
            grad = self.model.loss.gradient(self.get_cache('activation'))
            return grad * self.activation.derivative(self.get_cache('weighted_input'))

    def d_input(self):
        return np.matmul(self.error(), self.weight.T)

    def gradients(self):
        error = self.error()
        input_ = self.get_cache('input')
        error_stacked = np.reshape(error, (error.shape[0], 1,  error.shape[1]))
        input_stacked = np.reshape(input_, (input_.shape[0], input_.shape[1], 1))
        return {
            'bias': np.sum(error, axis=0),
            'weight': np.sum(np.matmul(input_stacked, error_stacked), axis=0) # np.sum(np.matmul(input, self.error()), axis=0)
        }

    def parameters(self):
        return {
            'bias': self.bias,
            'weight': self.weight
        }

    def forward(self, X, training=False):
        weighted = np.matmul(X, self.weight) + self.bias
        activation = self.activation.compute(weighted)
        if training:
            self.set_cache('weighted_input', weighted)
            self.set_cache('activation', activation)
            self.set_cache('input', X)
        return activation


class Flatten(Layer):
    def forward(self, inputs, training=False):
        result = np.reshape(inputs, (inputs.shape[0], -1))
        if training:
            self.set_cache('output', result)
        return result

    def set_previous(self, previous_layer):
        super().set_previous(previous_layer)
        self.set_output_shape()

    def set_output_shape(self):
        self.output_shape = (reduce(mul, self.input_shape),)

    @cached('error')
    def error(self):
        output = self.get_cache('output')
        return output * self.next_layer.d_input()

    def d_input(self):
        return np.reshape(self.error(), (-1,) + self.input_shape)


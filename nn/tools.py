import numpy as np
from scipy.stats import truncnorm


class NotYetSupportedError(Exception):
    pass


def filter2d(input_, filter_, stride=(1, 1)):
    """ Convolve (crosscorrelate, really) a filter with an input vector. """
    h_out = (input_.shape[1] - filter_.shape[0]) // stride[0] + 1
    v_out = (input_.shape[2] - filter_.shape[1]) // stride[1] + 1
    result = np.zeros((input_.shape[0], h_out, v_out, filter_.shape[-1]))
    for i in range(h_out):
        for j in range(v_out):
            result[:, i, j, :] = np.sum(input_[:, i * stride[0]:i * stride[0] + filter_.shape[0],
                                        j * stride[1]:j * stride[1] + filter_.shape[1], :,
                                        np.newaxis] * filter_, axis=(1, 2, 3))

    return result


def zero_pad(input_, filter_size, mode):
    """ Pad an input vector for convolution with a filter of the given size."""
    if mode == 'valid':
        return input_
    elif mode == 'same':
        l_pad, t_pad = (filter_size[0] - 1) // 2, (filter_size[1] - 1) // 2
        r_pad, b_pad = filter_size[0] - 1 - l_pad, filter_size[1] - 1 - t_pad
        return np.pad(input_, ((0, 0), (l_pad, r_pad), (t_pad, b_pad), (0, 0)), 'constant')
    elif mode == 'full':
        h_pad, v_pad = filter_size[0] - 1, filter_size[1] - 1
        return np.pad(input_, ((0, 0), (h_pad, h_pad), (v_pad, v_pad), (0, 0)), 'constant')


def truncated_normal(shape, mean=0, stddev=1):
    """ Generate values from a normal distribution truncated at 2 std deviations."""
    return stddev * (truncnorm.rvs(-2, 2, size=shape)) + mean


def softmax(x):
    """ Softmax of an array. """
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=1)[:, np.newaxis]

import numpy as np
from scipy.stats import truncnorm


def truncated_normal(shape, mean=0, stddev=1):
    return stddev * (truncnorm.rvs(-2, 2, size=shape) + mean)


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=1)[:, np.newaxis]


import numpy as np

class Activation:
    @staticmethod
    def compute(x):
        raise NotImplemented

    @staticmethod
    def derivative(x):
        raise NotImplemented

class ReLU(Activation):
  @staticmethod
  def compute(x):
      return np.maximum(x, 0)

  @staticmethod
  def derivative(x):
      return (x > 0).astype(np.float32)

class SoftMax(Activation):
  @staticmethod
  def compute(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp)

  @staticmethod
  def derivative(x):
      pass

class Identity(Activation):
    @staticmethod
    def compute(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(x.shape)
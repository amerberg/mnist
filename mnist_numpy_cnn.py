"""A convnet MNIST classifier, with architecture as dictated in the assignment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

from nn.layer import Conv2D, MaxPool2D, Flatten, Dense
from nn.model import Model
from nn.activation import ReLU
from nn.tools import truncated_normal
from nn.optimizer import AdaGrad, GradientDescent
from nn.loss import SoftMaxCrossEntropyWithLogits
FLAGS = None


def weight_initializer(shape):
    return truncated_normal(shape, stddev=0.1)


def bias_initializer(shape):
    return 0.1 * np.ones(shape)


def main(args):
    # Import data
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)
    relu = ReLU()

    model = Model(
      layers=[
          Conv2D(filter_size=(10, 10), input_shape=(28, 28, 1), stride=(1, 1), channels=32,
                 activation=relu, padding='same', filter_initializer=weight_initializer,
                 bias_initializer=bias_initializer),
          MaxPool2D(pool_size=(2, 2)),
          Conv2D(filter_size=(5, 5), stride=(1, 1), channels=16, activation=relu, padding='same',
                 filter_initializer=weight_initializer,
                 bias_initializer=bias_initializer),
          MaxPool2D(pool_size=(2, 2)),
          Flatten(),
          Dense(1024, weight_initializer=weight_initializer, bias_initializer=bias_initializer, activation=relu),
          Dense(10, weight_initializer=weight_initializer, bias_initializer=bias_initializer)
      ],
      optimizer=AdaGrad(learning_rate=0.001, epsilon=1e-8),
      loss=SoftMaxCrossEntropyWithLogits()
    )

    # Train
    for _ in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
        model.fit_batch(batch_xs, batch_ys)
        print(model.batch_number)
        print('Batch loss: {}'.format(model.loss.compute(model.predict(batch_xs), batch_ys)))

    # Test trained model
    actual_labels = np.argmax(mnist.test.labels[:1000, :], 1)
    predictions = model.predict(np.reshape(mnist.test.images[:1000, :], [-1, 28, 28, 1]))
    predicted_labels = np.argmax(predictions, 1)
    print(predicted_labels)

    accuracy = (actual_labels == predicted_labels).mean()
    print("Test accuracy: {}".format(accuracy))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  args = parser.parse_args()
  main(args)

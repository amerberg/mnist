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
from nn.tools import truncated_normal
from nn.optimizer import GradientDescent
from nn.loss import SoftMaxCrossEntropyWithLogits
FLAGS = None

def weight_initializer(shape):
    return truncated_normal(shape, stddev=0.1)

def bias_initializer(shape):
    return 0.1 * np.ones(shape)


def main(args):
    # Import data
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    model = Model(
      layers=[
          Dense(10, weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                input_shape=(784,))
      ],
      optimizer=GradientDescent(learning_rate=0.5),
      loss=SoftMaxCrossEntropyWithLogits()
    )

    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        model.fit_batch(batch_xs, batch_ys)

    # Test trained model
    actual_labels = np.argmax(mnist.test.labels, 1)
    predictions = model.predict(mnist.test.images)
    predicted_labels = np.argmax(predictions, 1)

    accuracy = (actual_labels == predicted_labels).mean()
    print("Test accuracy: {}".format(accuracy))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  args = parser.parse_args()
  main(args)

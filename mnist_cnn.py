"""A convnet MNIST classifier, with architecture as dictated in the assignment.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    W1 = tf.get_variable("W1", initializer=tf.truncated_normal([10, 10, 1, 32], 0, 0.1))
    b1 = tf.get_variable("b1", initializer=0.1 * tf.zeros([32]))
    W2 = tf.get_variable("W2", initializer=tf.truncated_normal([5, 5, 32, 16], 0, 0.1))
    b2 = tf.get_variable("b2", initializer=0.1 * tf.ones([16]))

    C1 = tf.nn.conv2d(input=x, filter=W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(C1 + b1)
    M1 = tf.nn.max_pool(value=A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    C2 = tf.nn.conv2d(input=M1, filter=W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(C2 + b2)
    M2 = tf.nn.max_pool(value=A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    flat = tf.reshape(M2, [-1, M2.shape[1:].num_elements()])

    W3 = tf.get_variable("W3", initializer=tf.truncated_normal([flat.shape[-1:].num_elements(), 1024], 0, 0.1))
    b3 = tf.get_variable("b3", initializer=0.1 * tf.ones([1024]))
    A3 = tf.nn.relu(tf.matmul(flat, W3) + b3)

    W4 = tf.get_variable("W4", initializer=tf.truncated_normal([1024, 10], 0, 0.1))
    b4 = tf.get_variable("b4", initializer=0.1 * tf.ones([10]))
    y = tf.matmul(A3, W4) + b4

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    # We use a learning rate of 0.01 after first trying 0.5 and 0.1 and getting much better accuracy with 0.01.
    train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Most of what's below is borrowed with minor modifications from the softmax example.
    # Train
    for _ in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: np.reshape(batch_xs, [-1, 28, 28, 1]), y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: np.reshape(mnist.test.images, [-1, 28, 28, 1]),
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

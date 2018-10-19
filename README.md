# MNIST digit classification
This repository implements some neural network classifiers for classification of the MNIST dataset:

- `mnist_softmax.py` is an example softmax classifier using TensorFlow. It is [an example from the Tensorflow repository](https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py), rather than my own work.
- `mnist_softmax_numpy.py` implements the same classifier as `mnist_softmax.py` using the `numpy`-based framework in `nn`
- `mnist_convnet.py` implements a convolutional neural network classifier
- `mnist_convnet_numpy.py` implements the same neural network architecture as `mnist_convnet.py` using the `numpy`-based framework in `nn` instead of TensorFlow. Performance is a bit worse (89% vs 94%), but I haven't yet figured out why.

# The `nn` classes
To define a model,  use `nn.model.Model`.
A model is defined by a list of layers, an optimizer, and a loss function.
One can use the `fit_batch` method to train the model on a single batch.
The `predict` method computes predictions once the model has been trained.

Layers are subclasses of `nn.layer.Layer`.
The base `Layer` has methods for caching, to allow values to be saved from the forward pass to the backward pass, or to prevent repeated calculations.
The following layers are included: `nn.layer.Conv2D`, `nn.layer.MaxPool2D`, `nn.layer.Flatten`, `nn.layer.Dense` (fully connected).
I hope that the names are mostly clear from context.
The constructor of the first layer in a network should be passed the `input_shape` parameter.
Other layers should be able to infer that information from context.

Some layers have additional parameters. 
The `Conv2D` layer requires a filter shape and the number of channels.
The `Dense` layer requires the size of the layer be passed to the constructor.
Both of these layers also allow the user to supply functions to initialize weights or filters and biases.
The initializer should take a single argument, a tuple of integers, and return an array with that shape.
The convolutional and fully connected layers can also take activation functions.

Some layers have parameter limitations because I haven't gotten around to implementing all values.
For instance, `MaxPool2D` does not allow padding and requires the stride to be the same as the pool size and the `Conv2D` layer only implements `same' padding.

Activation functions are subclasses of `nn.activation.Activation`.
The main activation included is `nn.activation.ReLU`, but there is also `nn.activation.Identity` which is the identity function, for use in layers for which no activation is desired.

Loss functions are subclasses of `nn.loss.Loss`.
The only loss function included in the module is the one that is called for by the assignment, `nn.loss.SoftMaxCrossEntropyWithLogits`.

Optimizers are subclasses of `nn.optimizer.Optimizer`.
An optimizer should implement the `update_layer` method to retrieve parameters and gradients from a layer and update them.
An optimizer may make use of the layer object's persistent cache to store data from one training batch to another.
The included optimizers are `nn.optimizer.GradientDescent` and `nn.optimizer.AdaGrad`.
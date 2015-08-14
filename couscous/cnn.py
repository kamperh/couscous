"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from theano.tensor import nnet
from theano.tensor.signal import downsample
import copy
import cPickle as pickle
import logging
import numpy as np
import scipy.signal
import theano
import theano.tensor as T

import logistic
import mlp
import theano_utils

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                       CONVOLUTIONAL MAX-POOLING CLASS                       #
#-----------------------------------------------------------------------------#

class ConvMaxPoolLayer(object):
    """
    Layer performing convolution, max-pooling and a non-linearity.

    Attributes
    ----------
    W : shared tensor of shape (n_out_filters, n_in_channels, height, width)
        The filter weights.
    b : shared vector of size n_out_filters
        The filter biases, one per output filter.
    output : symbolic expression
        The output of the convolutional, max-pooling layer with non-linearity.
    """

    def __init__(self, rng, input, input_shape, filter_shape,
            pool_shape=(2, 2), activation=T.tanh, W=None, b=None):
        """
        Initialize symbolic parameters and expressions.

        Parameters
        ----------
        rng : numpy.random.RandomState
            Random number generator used for weight initialization.
        input : symbolic tensor of shape (n_data, n_channels, height, width)
            Symbolic tensor describing the input (i.e. one minibatch).
        input_shape : (int, int, int, int)
            The shape of `input`: (n_data, n_channels, height, width).
        filter_shape : (int, int, int, int)
            Tuple of n_out_filters, n_in_channels, filter_height, filter_width.
        pool_shape : (int, int)
            Height and width of the pool window.
        """

        assert input_shape[1] == filter_shape[1]

        if W is None:
            n_units_in = np.prod(filter_shape[1:])
            n_units_out = np.prod(filter_shape[0] * np.prod(filter_shape[2:])) / np.prod(pool_shape)

            # Initialize weights
            if activation == T.tanh:
                W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_units_in + n_units_out)),
                    high=np.sqrt(6. / (n_units_in + n_units_out)),
                    size=filter_shape
                    ), dtype=theano.config.floatX)
            elif activation == theano.tensor.nnet.sigmoid:
                W_values = 4. * np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_units_in + n_units_out)),
                    high=np.sqrt(6. / (n_units_in + n_units_out)),
                    size=filter_shape
                    ), dtype=theano.config.floatX)
            elif activation == theano_utils.relu:
                W_values = np.asarray(0.01*rng.randn(*filter_shape), dtype=theano.config.floatX)
            elif activation is None:
                # Linear activation
                W_values = np.asarray(0.01*rng.randn(*filter_shape), dtype=theano.config.floatX)
            else:
                assert False, "Invalid activation"
            self.W = theano.shared(W_values, borrow=True)
        else:
            self.W = W

        # Initialize biases
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            # if activation == theano_utils.relu:
            #     b_values = 0.01*np.ones((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)
        self.b = b

        # Expression to convolve input with filters
        conv_out = nnet.conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=input_shape
            )

        # Expression to do max-pooling
        pool_out = downsample.max_pool_2d(
            input=conv_out, ds=pool_shape, ignore_border=True
            )

        # Expression of layer output
        linear_output = pool_out + self.b.dimshuffle("x", 0, "x", "x")
        self.output = (linear_output if activation is None else activation(linear_output))

        # self.output = theano.printing.Print("output")(self.output)

        self.parameters = [self.W, self.b]

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        pickle.dump(self.W.get_value(borrow=True), f, -1)
        pickle.dump(self.b.get_value(borrow=True), f, -1)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.W.set_value(pickle.load(f), borrow=True)
        self.b.set_value(pickle.load(f), borrow=True)

#-----------------------------------------------------------------------------#
#                   DROPOUT CONVOLUTIONAL MAX-POOLING CLASS                   #
#-----------------------------------------------------------------------------#

class DropoutConvMaxPoolLayer(ConvMaxPoolLayer):
    def __init__(self, rng, srng, dropout_rate, input, input_shape,
            filter_shape, pool_shape=(2, 2), activation=T.tanh, W=None,
            b=None):
        """
        Apart from the `srng` and `dropout_rate`, the parameters are
        identical to those of `ConvMaxPoolLayer`.
        """
        super(DropoutConvMaxPoolLayer, self).__init__(
            rng, input, input_shape, filter_shape, pool_shape, activation, W, b
            )
        self.output = theano_utils.apply_dropout(srng, self.output, p=dropout_rate)


#-----------------------------------------------------------------------------#
#                      CONVOLUTIONAL NEURAL NETWORK CLASS                     #
#-----------------------------------------------------------------------------#

class CNN(object):
    """
    Convolutional neural network.

    Attributes
    ----------
    layers : list of ConvMaxPoolLayer, HiddenLayer and LogisticRegression
        The convolutional and fully-connected hidden layers of the CNN, as well
        as the final logistic regression layer.
    """

    def __init__(self, rng, input, input_shape, conv_layer_specs,
            hidden_layer_specs, d_out, srng=None, dropout_rates=None):
        """
        Initialize symbolic parameters and expressions.

        Most of the parameters are identical to that of `build_cnn_layers`,
        which is used to build all the layers of the CNN, with a logistic
        regression layer added on top.

        Parameters
        ----------
        d_out : int
            Number of output classes.
        """

        self.input = input

        # Build convolutional and fully-connected hidden layers
        if dropout_rates is not None:
            self.dropout_layers, self.layers = build_cnn_layers(
                rng, input, input_shape, conv_layer_specs, hidden_layer_specs,
                srng, dropout_rates
                )
        else:
            self.layers = build_cnn_layers(
                rng, input, input_shape, conv_layer_specs,
                hidden_layer_specs
                )

        # Build logistic regression class prediction layer
        layer = logistic.LogisticRegression(
            input=self.layers[-1].output,
            d_in=hidden_layer_specs[-1]["units"],
            d_out=d_out
            )
        self.layers.append(layer)
        if dropout_rates is not None:
            dropout_layer = logistic.LogisticRegression(
                input=self.dropout_layers[-1].output,
                d_in=hidden_layer_specs[-1]["units"],
                d_out=d_out,
                W=layer.W,
                b=layer.b
                )
            self.dropout_layers.append(dropout_layer)

        # Model parameters
        self.parameters = []
        self.l1 = 0.
        self.l2 = 0.
        for layer in self.layers:
            self.parameters += layer.parameters
            self.l1 += abs(layer.W).sum()
            self.l2 += (layer.W**2).sum()

        # Symbolic expressions of cost and prediction
        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        if dropout_rates is not None:
            self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
            self.dropout_errors = self.dropout_layers[-1].errors

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        for layer in self.layers:
            layer.load(f)


#-----------------------------------------------------------------------------#
#                          NETWORK BUILDING FUNCTIONS                         #
#-----------------------------------------------------------------------------#

def build_cnn_layers(rng, input, input_shape, conv_layer_specs,
        hidden_layer_specs, srng=None, dropout_rates=None,
        init_W=None, init_b=None):
    """
    Return the layers of a CNN consisting of a number of convolutional layers
    followed by a number of fully-connected hidden layers.

    The convolutional layers are built according to `conv_layer_specs`, a list
    of dict which gives the specifications for each layer. Each dict has fields
    "filter_shape" and "pool_shape". The filter shapes are given as
    (n_out_filters, n_in_channels, filter_height, filter_width) while the pool
    shapes are (height, width). As an example, a network with single-channel
    (28, 28) shaped input images with 2 convolutional layers followed by 2
    fully-connected layers could be built using:

        batch_size = 10
        rng = np.random.RandomState(42)
        input = T.matrix("x")
        input_shape=(batch_size, 1, 28, 28)
        conv_layer_specs = [
            {
                "filter_shape": (20, 1, 5, 5), "pool_shape": (2, 2),
                "activation": theano_utils.relu
            }, 
            {
                "filter_shape": (50, 20, 5, 5), "pool_shape": (2, 2)},
                "activation": theano_utils.relu
            }
            ]
        hidden_layer_specs = [
            {"units": 500, "activation": theano_utils.relu},
            {"units": 500, "activation": theano_utils.relu}
            ]
        cnn_layers = build_cnn_layers(
            rng, input, input_shape, conv_layer_specs, hidden_layer_specs
            )

    Parameters
    ----------
    input : symbolic tensor
        Input to the first layer of the CNN. The first dimension should be
        across data instances.
    input_shape : (int, int, int, int)
        The shape of the input: (n_data, n_channels, height, width).
    conv_layer_specs : list of dict
        Specifications for the convolutional layers.
    hidden_layer_specs : list of dict
        Specifications for the fully-connected hidden layers.
    dropout_rates : list of float
        The dropout rates for each of the layers (including the convolutional
        layers); if not provided, dropout is not performed.
    init_W : list of shared tensors
        If provided, these weights are used for layer initialization. The
        weights should be given in the same order that the layers are created
        (i.e. first the convolutional weights and then the fully-connected
        hidden layer weights). This is useful for tying weights.
    init_b : list of shared vectors
        If provided, these biases are used for layer initialization. The order
        should be the same as that of `init_W`.
    """

    assert len(conv_layer_specs) > 0, "Use MLP class if no convolutional layers"
    assert (
        dropout_rates is None or 
        len(dropout_rates) == len(conv_layer_specs) + len(hidden_layer_specs)
        )

    conv_layer_specs = copy.deepcopy(conv_layer_specs)
    hidden_layer_specs = copy.deepcopy(hidden_layer_specs)
    for layer_spec in conv_layer_specs:
        mlp.activation_str_to_op(layer_spec)
    for layer_spec in hidden_layer_specs:
        mlp.activation_str_to_op(layer_spec)

    if init_W is not None:
        assert init_b is not None

        # We are going to pop parameters, so make copies
        init_W = init_W[:]
        init_b = init_b[:]

    layers = []
    if dropout_rates is not None:
        dropout_layers = []

    # Build convolutional layers

    for i_layer in xrange(len(conv_layer_specs)):
        if i_layer == 0:
            cur_input_shape = input_shape
            cur_input = input.reshape(input_shape)
        else:
            batch_size, prev_n_in_channels, prev_in_height, prev_in_width = prev_input_shape
            prev_n_out_filters, prev_n_in_channels, prev_filter_height, prev_filter_width = (
                prev_filter_shape
                )
            prev_pool_height, prev_pool_width = prev_pool_shape
            cur_input_shape = (
                batch_size,
                prev_n_out_filters,
                int(np.floor(1. * (prev_in_height - prev_filter_height + 1) / prev_pool_height)),
                int(np.floor(1. * (prev_in_width - prev_filter_width + 1) / prev_pool_width))
                )
            cur_input = layers[-1].output

        if init_W is not None:
            W = init_W.pop(0)
            b = init_b.pop(0)
        else:
            W = None
            b = None
        cur_activation = conv_layer_specs[i_layer]["activation"]

        layer = ConvMaxPoolLayer(
            rng,
            input=cur_input,
            input_shape=cur_input_shape,
            filter_shape=conv_layer_specs[i_layer]["filter_shape"],
            pool_shape=conv_layer_specs[i_layer]["pool_shape"],
            activation=cur_activation,
            W=W,
            b=b
            )
        layers.append(layer)

        if dropout_rates is not None:
            if i_layer == 0:
                cur_dropout_input = input.reshape(input_shape)
            else:
                cur_dropout_input = dropout_layers[-1].output
            dropout_rate = dropout_rates[i_layer]
            dropout_layer = DropoutConvMaxPoolLayer(
                rng, srng, dropout_rate,
                input=cur_dropout_input,
                input_shape=cur_input_shape,
                filter_shape=conv_layer_specs[i_layer]["filter_shape"],
                pool_shape=conv_layer_specs[i_layer]["pool_shape"],
                activation=cur_activation,
                W=layer.W / (1. - dropout_rate),
                b=layer.b
                )
            dropout_layers.append(dropout_layer)

        # Store shapes for next layer
        prev_input_shape = cur_input_shape
        prev_filter_shape = conv_layer_specs[i_layer]["filter_shape"]
        prev_pool_shape = conv_layer_specs[i_layer]["pool_shape"]

    # Build fully-connected hidden layers

    for i_layer in xrange(len(hidden_layer_specs)):

        if i_layer == 0:
            # Shapes from last convolutional layer
            batch_size, prev_n_in_channels, prev_in_height, prev_in_width = prev_input_shape
            prev_n_out_filters, prev_n_in_channels, prev_filter_height, prev_filter_width = (
                prev_filter_shape
                )
            prev_pool_height, prev_pool_width = prev_pool_shape
            cur_d_in = (
                prev_n_out_filters *
                int(np.floor(1. * (prev_in_height - prev_filter_height + 1) / prev_pool_height)) *
                int(np.floor(1. * (prev_in_width - prev_filter_width + 1) / prev_pool_width))
                )
            cur_input = layers[-1].output.flatten(2)
        else:
            cur_d_in = hidden_layer_specs[i_layer - 1]["units"]
            cur_input = layers[-1].output

        if init_W is not None:
            W = init_W.pop(0)
            b = init_b.pop(0)
        else:
            W = None
            b = None
        cur_activation = hidden_layer_specs[i_layer]["activation"]
        layer = mlp.HiddenLayer(
            rng=rng,
            input=cur_input,
            d_in=cur_d_in,
            d_out=hidden_layer_specs[i_layer]["units"],
            activation=cur_activation,
            W=W,
            b=b
            )
        layers.append(layer)

        if dropout_rates is not None:
            if i_layer == 0:
                cur_dropout_input = dropout_layers[-1].output.flatten(2)
            else:
                cur_dropout_input = dropout_layers[-1].output
            dropout_rate = dropout_rates[len(conv_layer_specs) + i_layer]
            dropout_layer = mlp.DropoutHiddenLayer(
                rng=rng,
                srng=srng,
                dropout_rate=dropout_rate,
                input=cur_dropout_input,
                d_in=cur_d_in,
                d_out=hidden_layer_specs[i_layer]["units"],
                activation=cur_activation,
                W=layer.W / (1. - dropout_rate),
                b=layer.b
                )
            dropout_layers.append(dropout_layer)

    if dropout_rates is not None:
        return (dropout_layers, layers)
    return layers


#-----------------------------------------------------------------------------#
#                                TEST FUNCTIONS                               #
#-----------------------------------------------------------------------------#

def np_convolve_3d(input, filters, mode="valid"):
    """
    Calculate the 3-dimensional convolution of data `input` using weights `filters`.

    Parameters
    ----------
    input : matrix of shape (d_in, height, width)
        The data to be convolved.
    filters : matrix of shape (d_in, filter_height, filter_width)
        The filter weights.

    Return
    ------
    conv_result : matrix of shape (height_conv, width_conv)
    """
    d_in = filters.shape[0]
    conv_result = scipy.signal.convolve2d(filters[0, :, :], input[0, :, :], mode=mode)
    for i_channel in xrange(1, d_in):
        conv_result += scipy.signal.convolve2d(filters[i_channel, :, :], input[i_channel, :, :], mode=mode)
    return conv_result


def np_convolve_4d(input, filters, mode="valid"):
    """
    Calculate the 4-dimensional convolution of data `input` using weights `filters`.

    Parameters
    ----------
    input : matrix of shape (n_data, d_in, height, width)
        The data to be convolved.
    filters : matrix of shape (d_out, d_in, filter_height, filter_width)
        The filter weights.

    Return
    ------
    conv_over_data : matrix of shape (n_data, d_out, height_conv, width_conv)
    """
    n_data, d_in, height, width = input.shape
    d_out, _, filter_height, filter_width = filters.shape
    assert d_in == _

    # conv_result = np.zeros((n_data, d_out, height, width))
    # print conv_result.shape

    # Loop over data points
    conv_over_data = []
    for i_data in xrange(n_data):
        # Loop over channels
        conv_over_channels = []
        for i_out_channel in xrange(d_out):
            conv_over_channels.append(np_convolve_3d(filters[i_out_channel, :, :, :], input[i_data, :, :, :]))
        conv_over_data.append(np.array(conv_over_channels))
    return np.array(conv_over_data)


def np_max_pool_2d(input, size, ignore_border=False):
    """The `input` is in the same format as in `np_convolve_4d`."""

    if len(input.shape) == 2:
        input = input.copy().reshape(1, 1, input.shape[0], input.shape[1])

    # Dimensionality
    n_data, d_in, input_h, input_w = input.shape
    filter_h, filter_w = size
    round_func = np.floor if ignore_border else np.ceil
    output_h = int(round_func(1.*input_h/filter_h))
    output_w = int(round_func(1.*input_w/filter_w))

    # Find max
    max_result = np.zeros((n_data, d_in, output_h, output_w))
    for i_data in xrange(n_data):
        for i_channel in xrange(d_in):
            for i in xrange(output_h):
                for j in xrange(output_w):
                    max_result[i_data, i_channel, i, j] = np.max(input[
                        i_data, i_channel,
                        i*filter_h:i*filter_h + filter_h,
                        j*filter_w:j*filter_w + filter_w
                        ])

    return max_result


def np_cnn_layers_output(input, conv_layers_W, conv_layers_b,
        conv_layers_pool_shape, hidden_layers_W, hidden_layers_b,
        activation=np.tanh):
    batch_size = input.shape[0]
    np_output = input
    for W, b, pool_shape in zip(conv_layers_W, conv_layers_b, conv_layers_pool_shape):
        np_output = np_max_pool_2d(
            activation(np_convolve_4d(np_output, W) + b.reshape(1, b.shape[0], 1, 1)),
            size=pool_shape, ignore_border=True
            )
    for W, b in zip(hidden_layers_W, hidden_layers_b):
        np_output = activation(np.dot(np_output.reshape(batch_size, -1), W) + b)
    return np_output


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import numpy.testing as npt


    # Test `build_cnn_layers` with dropout

    # Random number generators
    from theano.tensor.shared_randomstreams import RandomStreams
    rng = np.random.RandomState(42)
    srng = RandomStreams(seed=42)

    # Generate random data
    n_data = 2
    height = 39
    width = 200
    in_channels = 1
    X = rng.randn(n_data, in_channels, height, width)

    # Setup Theano model
    batch_size = n_data
    input = T.matrix("x")
    input_shape = (batch_size, in_channels, height, width)
    conv_layer_specs = [
        {"filter_shape": (32, in_channels, 13, 9), "pool_shape": (3, 3), "activation": T.tanh},
        {"filter_shape": (10, 32, 5, 5), "pool_shape": (3, 3), "activation": T.tanh}, 
        ]
    hidden_layer_specs = [
        {"units": 10, "activation": T.tanh},
        {"units": 10, "activation": T.tanh}
        ]
    dropout_rates = [0.0, 0.1, 0.2, 0.3]
    dropout_cnn_layers, cnn_layers = build_cnn_layers(
        rng, input, input_shape, conv_layer_specs, hidden_layer_specs,
        srng=srng, dropout_rates=dropout_rates
        )

    # Compile Theano function
    theano_dropout_cnn_layers_output = theano.function(
        inputs=[input], outputs=dropout_cnn_layers[-1].output
        )
    theano_output = theano_dropout_cnn_layers_output(X.reshape(n_data, -1))
    print theano_output

    # Get weights in order to calculate Numpy output
    conv_layers_W = []
    conv_layers_b = []
    conv_layers_pool_shape = []
    hidden_layers_W = []
    hidden_layers_b = []
    for i_layer in xrange(len(conv_layer_specs)):
        W = cnn_layers[i_layer].W.get_value(borrow=True)
        b = cnn_layers[i_layer].b.get_value(borrow=True)
        pool_shape = conv_layer_specs[i_layer]["pool_shape"]
        conv_layers_W.append(W)
        conv_layers_b.append(b)
        conv_layers_pool_shape.append(pool_shape)
    for i_layer in xrange(i_layer + 1, i_layer + 1 + len(hidden_layer_specs)):
        W = cnn_layers[i_layer].W.get_value(borrow=True)
        b = cnn_layers[i_layer].b.get_value(borrow=True)
        hidden_layers_W.append(W)
        hidden_layers_b.append(b)

    # Calculate Numpy output
    srng = RandomStreams(seed=42)
    activation = np.tanh
    input = X
    batch_size = input.shape[0]
    np_output = input
    for W, b, pool_shape in zip(conv_layers_W, conv_layers_b, conv_layers_pool_shape):
        dropout_rate = dropout_rates.pop(0)
        W = W / (1. - dropout_rate)
        np_output = np_max_pool_2d(
            activation(np_convolve_4d(np_output, W) + b.reshape(1, b.shape[0], 1, 1)),
            size=pool_shape, ignore_border=True
            )
        np_output = theano_utils.np_apply_dropout(srng, np_output, dropout_rate)
    for W, b in zip(hidden_layers_W, hidden_layers_b):
        dropout_rate = dropout_rates.pop(0)
        W = W / (1. - dropout_rate)
        np_output = activation(np.dot(np_output.reshape(batch_size, -1), W) + b)
        np_output = theano_utils.np_apply_dropout(srng, np_output, dropout_rate)

    print np_output

    npt.assert_almost_equal(np_output, theano_output)


    # # Test `build_cnn_layers`

    # # Random number generators
    # from theano.tensor.shared_randomstreams import RandomStreams
    # rng = np.random.RandomState(42)
    # srng = RandomStreams(seed=42)

    # # Generate random data
    # n_data = 2
    # height = 39
    # width = 200
    # in_channels = 1
    # X = rng.randn(n_data, in_channels, height, width)

    # # Setup Theano model
    # batch_size = n_data
    # input = T.matrix("x")
    # input_shape = (batch_size, in_channels, height, width)
    # conv_layer_specs = [
    #     {"filter_shape": (32, in_channels, 13, 9), "pool_shape": (3, 3), "activation": theano_utils.relu}, 
    #     {"filter_shape": (10, 32, 5, 5), "pool_shape": (3, 3), "activation": theano_utils.relu}, 
    #     ]
    # hidden_layer_specs = [
    #     {"units": 128, "activation": theano_utils.relu},
    #     {"units": 10, "activation": theano_utils.relu}
    #     ]
    # cnn_layers = build_cnn_layers(
    #     rng, input, input_shape, conv_layer_specs, hidden_layer_specs
    #     )

    # # Compile Theano function
    # theano_cnn_layers_output = theano.function(
    #     inputs=[input], outputs=cnn_layers[-1].output
    #     )
    # theano_output = theano_cnn_layers_output(X.reshape(n_data, -1))
    # print theano_output

    # # Calculate Numpy output
    # conv_layers_W = []
    # conv_layers_b = []
    # conv_layers_pool_shape = []
    # hidden_layers_W = []
    # hidden_layers_b = []
    # for i_layer in xrange(len(conv_layer_specs)):
    #     W = cnn_layers[i_layer].W.get_value(borrow=True)
    #     b = cnn_layers[i_layer].b.get_value(borrow=True)
    #     pool_shape = conv_layer_specs[i_layer]["pool_shape"]
    #     conv_layers_W.append(W)
    #     conv_layers_b.append(b)
    #     conv_layers_pool_shape.append(pool_shape)
    # for i_layer in xrange(i_layer + 1, i_layer + 1 + len(hidden_layer_specs)):
    #     W = cnn_layers[i_layer].W.get_value(borrow=True)
    #     b = cnn_layers[i_layer].b.get_value(borrow=True)
    #     hidden_layers_W.append(W)
    #     hidden_layers_b.append(b)
    # np_output = np_cnn_layers_output(
    #     X, conv_layers_W, conv_layers_b, conv_layers_pool_shape,
    #     hidden_layers_W, hidden_layers_b, activation=theano_utils.np_relu
    #     )
    # print np_output

    # npt.assert_almost_equal(np_output, theano_output)
    
    # # Test `ConvMaxPoolLayer`

    # # Test setup
    # n_data = 2
    # n_in_channels = 3
    # width = 28
    # height = 28
    # n_out_filters = 3
    # filter_height = 25
    # filter_width = 25
    # pool_shape = (2, 3)

    # # Setup model
    # rng = np.random.RandomState(42)
    # x = T.tensor4("x")  # shape: (n_data, n_in_channels, width, height)
    # # x_reshaped = x.reshape((n_data, n_in_channels, width, height))
    # conv_layer = ConvMaxPoolLayer(
    #     rng,
    #     input=x,
    #     input_shape=(n_data, n_in_channels, height, width),
    #     filter_shape=(n_out_filters, n_in_channels, filter_height, filter_width),
    #     pool_shape=pool_shape
    #     )
    # conv_output = conv_layer.output

    # # Generate non-zero biases
    # b_test = rng.randn(n_out_filters)
    # conv_layer.b.set_value(b_test)

    # # Compile a test function
    # test_layer = theano.function(inputs=[x], outputs=conv_output)

    # # Get weights and biases from model
    # W = conv_layer.W.get_value()
    # b = conv_layer.b.get_value()

    # # Generate random data
    # X = rng.randn(n_data, n_in_channels, width, height)

    # # Calculate Numpy output
    # np_conv_output = np_max_pool_2d(
    #     np.tanh(np_convolve_4d(X, W) + b.reshape(1, b.shape[0], 1, 1)),
    #     size=pool_shape, ignore_border=True
    #     )
    # print "Numpy output:\n", np_conv_output

    # theano_conv_output = test_layer(X)
    # print "Theano output:\n", theano_conv_output

    # npt.assert_almost_equal(np_conv_output, theano_conv_output)


if __name__ == "__main__":
    main()

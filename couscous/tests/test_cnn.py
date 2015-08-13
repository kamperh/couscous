"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import numpy as np
import numpy.testing as npt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from couscous import theano_utils
from couscous.cnn import (
    ConvMaxPoolLayer, build_cnn_layers, np_cnn_layers_output, np_convolve_4d,
    np_max_pool_2d
    )


def test_conv_max_pool_layer():

    # Test setup
    n_data = 2
    n_in_channels = 3
    width = 28
    height = 28
    n_out_filters = 3
    filter_height = 25
    filter_width = 25
    pool_shape = (2, 3)

    # Setup model
    rng = np.random.RandomState(42)
    x = T.matrix("x")
    x = x.reshape((n_data, n_in_channels, width, height))
    conv_layer = ConvMaxPoolLayer(
        rng,
        input=x,
        input_shape=(n_data, n_in_channels, height, width),
        filter_shape=(n_out_filters, n_in_channels, filter_height, filter_width),
        pool_shape=pool_shape
        )
    conv_output = conv_layer.output

    # Generate non-zero biases
    b_test = np.asarray(rng.randn(n_out_filters), dtype=theano.config.floatX)
    conv_layer.b.set_value(b_test)

    # Compile a test function
    test_layer = theano.function(inputs=[x], outputs=conv_output)

    # Get weights and biases from model
    W = conv_layer.W.get_value()
    b = conv_layer.b.get_value()

    # Generate random data
    X = np.asarray(rng.randn(n_data, n_in_channels, width, height), dtype=theano.config.floatX)

    # Calculate Numpy output
    np_conv_output = np_max_pool_2d(
        np.tanh(np_convolve_4d(X, W) + b.reshape(1, b.shape[0], 1, 1)),
        size=pool_shape, ignore_border=True
        )
    # print "Numpy output:\n", np_conv_output

    theano_conv_output = test_layer(X)
    # print "Theano output:\n", theano_conv_output

    npt.assert_almost_equal(np_conv_output, theano_conv_output, decimal=6)


def test_cnn_build_layers():

    rng = np.random.RandomState(42)
    srng = RandomStreams(seed=42)

    # Generate random data
    n_data = 5
    height = 39
    width = 100
    in_channels = 2
    X = rng.randn(n_data, in_channels, height, width)

    # Setup theano model
    batch_size = n_data
    input = T.matrix("x")
    input_shape = (batch_size, in_channels, height, width)
    conv_layer_specs = [
        {"filter_shape": (32, in_channels, 13, 9), "pool_shape": (3, 3)}, 
        {"filter_shape": (10, 32, 5, 5), "pool_shape": (3, 3)}, 
        ]
    hidden_layer_specs = [{"units": 128}, {"units": 10}]
    cnn_layers = build_cnn_layers(
        rng, input, input_shape, conv_layer_specs, hidden_layer_specs
        )

    # Compile theano function
    theano_cnn_layers_output = theano.function(
        inputs=[input], outputs=cnn_layers[-1].output
        )
    theano_output = theano_cnn_layers_output(X.reshape(n_data, -1))
    # print theano_output

    # Calculate Numpy output
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
    np_output = np_cnn_layers_output(
        X, conv_layers_W, conv_layers_b, conv_layers_pool_shape,
        hidden_layers_W, hidden_layers_b
        )
    # print np_output

    npt.assert_almost_equal(np_output, theano_output)



def test_cnn_build_layers_dropout():

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
    # print theano_output

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

    # print np_output

    npt.assert_almost_equal(np_output, theano_output)

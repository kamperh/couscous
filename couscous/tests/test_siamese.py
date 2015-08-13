"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from theano.tensor.shared_randomstreams import RandomStreams
import itertools
import numpy as np
import numpy.testing as npt
import scipy.spatial.distance as distance
import theano
import theano.tensor as T

from couscous import theano_utils
from couscous.cnn import np_cnn_layers_output
from couscous.siamese import SiameseCNN, np_loss_cos_cos2


def test_siamese_cnn():

    # Random number generators
    rng = np.random.RandomState(42)
    srng = RandomStreams(seed=42)

    # Generate random data
    n_data = 4
    n_pairs = 6
    height = 39
    width = 200
    in_channels = 1
    X = rng.randn(n_data, in_channels, height, width)
    Y = np.asarray(rng.randint(2, size=n_pairs), dtype=np.int32)
    print "Same/diff:", Y

    # Generate random pairs
    possible_pairs = list(itertools.combinations(range(n_data), 2))
    x1_indices = []
    x2_indices = []
    for i_pair in rng.choice(np.arange(len(possible_pairs)), size=n_pairs, replace=False):
        x1, x2 = possible_pairs[i_pair]
        x1_indices.append(x1)
        x2_indices.append(x2)
    x1_indices = np.array(x1_indices)
    x2_indices = np.array(x2_indices)
    print "x1 index: ", x1_indices
    print "x2 index: ", x2_indices

    # Setup Theano model
    batch_size = n_pairs
    input_shape = (batch_size, in_channels, height, width)
    conv_layer_specs = [
        {"filter_shape": (32, in_channels, 39, 9), "pool_shape": (1, 3)},
        ]
    hidden_layer_specs = [{"units": 128}]
    dropout_rates = None
    y = T.ivector("y")
    input_x1 = T.matrix("x1")
    input_x2 = T.matrix("x2")
    model = SiameseCNN(
            rng, input_x1, input_x2, input_shape,
            conv_layer_specs, hidden_layer_specs, srng,
            dropout_rates=dropout_rates,
        )
    loss = model.loss_cos_cos2(y)

    # Compile Theano function
    theano_siamese_loss = theano.function(
        inputs=[], outputs=loss,
        givens={
            input_x1: X.reshape((n_data, -1))[x1_indices],
            input_x2: X.reshape((n_data, -1))[x2_indices],
            y: Y
            },
        )
    theano_loss = theano_siamese_loss()
    print "Theano loss:", theano_loss

    # Calculate Numpy output
    conv_layers_W = []
    conv_layers_b = []
    conv_layers_pool_shape = []
    hidden_layers_W = []
    hidden_layers_b = []
    for i_layer in xrange(len(conv_layer_specs)):
        W = model.layers[i_layer].W.get_value(borrow=True)
        b = model.layers[i_layer].b.get_value(borrow=True)
        pool_shape = conv_layer_specs[i_layer]["pool_shape"]
        conv_layers_W.append(W)
        conv_layers_b.append(b)
        conv_layers_pool_shape.append(pool_shape)
    for i_layer in xrange(i_layer + 1, i_layer + 1 + len(hidden_layer_specs)):
        W = model.layers[i_layer].W.get_value(borrow=True)
        b = model.layers[i_layer].b.get_value(borrow=True)
        hidden_layers_W.append(W)
        hidden_layers_b.append(b)
    np_x1_layers_output = np_cnn_layers_output(
        X[x1_indices], conv_layers_W, conv_layers_b, conv_layers_pool_shape,
        hidden_layers_W, hidden_layers_b
        )
    np_x2_layers_output = np_cnn_layers_output(
        X[x2_indices], conv_layers_W, conv_layers_b, conv_layers_pool_shape,
        hidden_layers_W, hidden_layers_b
        )

    numpy_loss = np_loss_cos_cos2(np_x1_layers_output, np_x2_layers_output, Y)
    print "Numpy loss:", numpy_loss

    npt.assert_almost_equal(numpy_loss, theano_loss)

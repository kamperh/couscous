"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import numpy as np
import numpy.testing as npt
import theano
import theano.tensor as T

from couscous import theano_utils
from couscous.mlp import MLP, np_mlp_negative_log_likelihood


def test_mlp():

    # Setup model
    rng = np.random.RandomState(42)
    d_in = 20
    d_out = 16
    l1_weight=0.01
    l2_weight=0.0001
    hidden_layer_sizes = [20, 30, 20]
    x = T.matrix("x")       # input: feature vectors
    y = T.ivector("y")      # output: int values for each class
    model = MLP(rng, input=x, d_in=d_in, d_out=d_out, hidden_layer_sizes=hidden_layer_sizes)
    cost = model.negative_log_likelihood(y) + l1_weight * model.l1 + l2_weight * model.l2

    # Generate non-zero weights and biases and assign to logistic regression
    W_test = np.asarray(rng.randn(hidden_layer_sizes[-1], d_out), dtype=theano.config.floatX)
    b_test = np.asarray(rng.randn(d_out), dtype=theano.config.floatX)
    model.layers[-1].W.set_value(W_test)
    model.layers[-1].b.set_value(b_test)

    # Compile a test function
    test_model = theano.function(inputs=[x, y], outputs=cost)

    # Get the weights and biases after initialization
    l1 = 0.
    l2 = 0.
    hidden_layers_W = []
    hidden_layers_b = []
    for hidden_layer in model.layers[:-1]:
        cur_W = hidden_layer.W.get_value()
        cur_b = hidden_layer.b.get_value()
        hidden_layers_W.append(cur_W)
        hidden_layers_b.append(cur_b)
        l1 += abs(cur_W).sum()
        l2 += (cur_W**2).sum()
    logistic_regression_W = model.layers[-1].W.get_value()
    logistic_regression_b = model.layers[-1].b.get_value()
    l1 += abs(logistic_regression_W).sum()
    l2 += (logistic_regression_W**2).sum()

    # Generate random data
    N = 15
    X = np.asarray(rng.randn(N, d_in), dtype=theano.config.floatX)
    y = np.random.randint(d_out, size=N)
    y = y.astype("int32")

    # Calculate Numpy cost
    np_cost = np_mlp_negative_log_likelihood(
        X, y, hidden_layers_W, hidden_layers_b, logistic_regression_W,
        logistic_regression_b
        ) + l1_weight*l1 + l2_weight*l2
    print "Numpy cost:", np_cost

    # Calculate Theano cost
    theano_cost = test_model(X, y)
    print "Theano cost:", theano_cost

    npt.assert_almost_equal(theano_cost, np_cost, decimal=6)


def test_mlp_relu():

    # Setup model
    rng = np.random.RandomState(42)
    d_in = 20
    d_out = 16
    l1_weight=0.01
    l2_weight=0.0001
    hidden_layer_sizes = [20, 30, 20]
    hidden_layer_activation = theano_utils.relu
    x = T.matrix("x")       # input: feature vectors
    y = T.ivector("y")      # output: int values for each class
    model = MLP(
        rng, input=x, d_in=d_in, d_out=d_out,
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_layer_activation=hidden_layer_activation
        )
    cost = model.negative_log_likelihood(y) + l1_weight * model.l1 + l2_weight * model.l2

    # Generate non-zero weights and biases and assign to logistic regression
    W_test = np.asarray(rng.randn(hidden_layer_sizes[-1], d_out), dtype=theano.config.floatX)
    b_test = np.asarray(rng.randn(d_out), dtype=theano.config.floatX)
    model.layers[-1].W.set_value(W_test)
    model.layers[-1].b.set_value(b_test)

    # Compile a test function
    test_model = theano.function(inputs=[x, y], outputs=cost)

    # Get the weights and biases after initialization
    l1 = 0.
    l2 = 0.
    hidden_layers_W = []
    hidden_layers_b = []
    for hidden_layer in model.layers[:-1]:
        cur_W = hidden_layer.W.get_value()
        cur_b = hidden_layer.b.get_value()
        hidden_layers_W.append(cur_W)
        hidden_layers_b.append(cur_b)
        l1 += abs(cur_W).sum()
        l2 += (cur_W**2).sum()
    logistic_regression_W = model.layers[-1].W.get_value()
    logistic_regression_b = model.layers[-1].b.get_value()
    l1 += abs(logistic_regression_W).sum()
    l2 += (logistic_regression_W**2).sum()

    # Generate random data
    N = 15
    X = np.asarray(rng.randn(N, d_in), dtype=theano.config.floatX)
    y = np.random.randint(d_out, size=N)
    y = y.astype("int32")

    # Calculate Numpy cost
    np_cost = np_mlp_negative_log_likelihood(
        X, y, hidden_layers_W, hidden_layers_b, logistic_regression_W,
        logistic_regression_b, hidden_layer_activation=theano_utils.np_relu
        ) + l1_weight*l1 + l2_weight*l2
    print "Numpy cost:", np_cost

    # Calculate Theano cost
    theano_cost = test_model(X, y)
    print "Theano cost:", theano_cost

    npt.assert_almost_equal(theano_cost, np_cost, decimal=6)

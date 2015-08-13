"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import numpy as np
import numpy.testing as npt
import theano
import theano.tensor as T

from couscous.logistic import LogisticRegression, np_negative_log_likelihood


def test_logistic_regression():

    np.random.seed(42)

    # Setup model
    d_in = 20
    d_out = 15
    x = T.matrix("x")       # input: feature vectors
    y = T.ivector("y")      # output: int values for each class
    model = LogisticRegression(input=x, d_in=d_in, d_out=d_out)
    cost = model.negative_log_likelihood(y)

    # Generate non-zero weights and biases and assign to model
    W_test = np.asarray(np.random.randn(d_in, d_out), dtype=theano.config.floatX)
    b_test = np.asarray(np.random.randn(d_out), dtype=theano.config.floatX)
    model.W.set_value(W_test)
    model.b.set_value(b_test)

    # Compile a test function
    test_model = theano.function(inputs=[x, y], outputs=cost)

    # Generate random data
    N = 2
    X = np.asarray(np.random.randn(N, d_in), dtype=theano.config.floatX)
    y = np.random.randint(d_out, size=N)
    y = y.astype("int32")

    # Calculate Numpy cost
    np_cost = np_negative_log_likelihood(X, y, W_test, b_test)
    print "Numpy cost:", np_cost

    # Calculate Theano cost
    theano_cost = test_model(X, y)
    print "Theano cost:", theano_cost

    npt.assert_almost_equal(theano_cost, np_cost, decimal=6)

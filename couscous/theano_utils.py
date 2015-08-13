"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import numpy as np
import theano
import theano.tensor as T


def relu(x):
    return T.maximum(0., x)


def np_relu(x):
    return np.maximum(0., x)


def apply_dropout(srng, input, p):
    """The probablity of dropping a unit is given by `p`."""
    mask = srng.binomial(n=1, p=1 - p, size=input.shape)

    # input = theano.printing.Print("input")(input)
    # mask = theano.printing.Print("mask")(mask)

    # The cast is important for the GPU since int * float32 = float64
    output = input * T.cast(mask, theano.config.floatX)
    return output


def np_apply_dropout(srng, input, p):
    """The probablity of dropping a unit is given by `p`."""
    mask = srng.binomial(n=1, p=1 - p, size=input.shape)
    # The cast is important for the GPU since int * float32 = float64
    f = theano.function([], mask)
    output = input * f()
    return output

"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import numpy as np
import scipy.spatial.distance as distance
import theano
import theano.tensor as T

import cnn


#-----------------------------------------------------------------------------#
#                              SIAMESE CNN CLASS                              #
#-----------------------------------------------------------------------------#

class SiameseCNN(object):
    """
    Siamese convolutional neural network.

    Attributes
    ----------
    layers : list of ConvMaxPoolLayer and HiddenLayer
        The layers that are shared by the two sides of the Siamese network.
    dropout_layers : list of ConvMaxPoolLayer and HiddenLayer
        The same as the above, but with dropout applied.
    x1_layers : list of ConvMaxPoolLayer and HiddenLayer
        The one side of the network. Similarly, there is also `x2_layers`, and
        `x1_dropout_layers` and `x2_dropout_layers` when dropout is applied.
    """

    def __init__(self, rng, input_x1, input_x2, input_shape,
            conv_layer_specs, hidden_layer_specs, srng=None,
            dropout_rates=None):
        """
        Initialize symbolic parameters and expressions.

        Many of the parameters are identical to that of `cnn.build_cnn_layers`.
        Some of the other parameters are described below.

        Parameters
        ----------
        input_x1 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of one side of the Siamese network.
        input_x2 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the other side of the Siamese network.
        """

        # Build common layers to which the Siamese layers are tied
        input = T.matrix("x")
        self.input = input
        layers = cnn.build_cnn_layers(
            rng, input, input_shape, conv_layer_specs,
            hidden_layer_specs, srng, dropout_rates,
            )

        # Copy the shared variables
        shared_W = []
        shared_b = []
        # The isinstance deals with dropout (where `layers` is a tuple)
        for layer in (layers[1] if isinstance(layers, tuple) else layers):
            shared_W.append(layer.W)
            shared_b.append(layer.b)

        # Build the Siamese layers and tie their weights
        x1_layers = cnn.build_cnn_layers(
            rng, input_x1, input_shape, conv_layer_specs,
            hidden_layer_specs, srng, dropout_rates,
            init_W=shared_W, init_b=shared_b
            )
        x2_layers = cnn.build_cnn_layers(
            rng, input_x2, input_shape, conv_layer_specs,
            hidden_layer_specs, srng, dropout_rates,
            init_W=shared_W, init_b=shared_b
            )

        if dropout_rates is not None:
            # All the layers so far is actually tuples which includes the dropout layers
            self.dropout_layers, self.layers = layers
            self.x1_dropout_layers, self.x1_layers = x1_layers
            self.x2_dropout_layers, self.x2_layers = x2_layers
        else:
            self.layers = layers
            self.x1_layers = x1_layers
            self.x2_layers = x2_layers

        # Model parameters
        self.parameters = []
        self.l1 = 0.
        self.l2 = 0.
        for layer in self.layers:
            self.parameters += layer.parameters
            self.l1 += abs(layer.W).sum()
            self.l2 += (layer.W**2).sum()

    def loss_cos_cos2(self, y):
        """
        Return symbolic loss expression.

        Parameters
        ----------
        y : symbolic vector of size n_data
            Symbolic vector indicating when a pair is the same (1), meaning we
            would want to minimize a distance, or different (0), meaning we 
            would want to maximize a distance. All member loss functions have
            only this parameter.
        """
        return _loss_cos_cos2(
            self.x1_layers[-1].output, self.x2_layers[-1].output, y
            )

    def dropout_loss_cos_cos2(self, y):
        return _loss_cos_cos2(
            self.x1_dropout_layers[-1].output, self.x2_dropout_layers[-1].output, y
            )

    def loss_cos_cos(self, y):
        return _loss_cos_cos(
            self.x1_layers[-1].output, self.x2_layers[-1].output, y
            )

    def dropout_loss_cos_cos(self, y):
        return _loss_cos_cos(
            self.x1_dropout_layers[-1].output, self.x2_dropout_layers[-1].output, y
            )

    def loss_cos2(self, y):
        pass        

    def loss_cos_cos_margin(self, y, margin=0.5):
        return _loss_cos_cos_margin(
            self.x1_layers[-1].output, self.x2_layers[-1].output, y, margin
            )

    def dropout_loss_cos_cos_margin(self, y, margin=0.5):
        return _loss_cos_cos_margin(
            self.x1_dropout_layers[-1].output, self.x2_dropout_layers[-1].output, y, margin
            )

    def loss_euclidean_margin(self, y, margin=1):
        return _loss_euclidean_margin(
            self.x1_layers[-1].output, self.x2_layers[-1].output, y, margin
            )

    def dropout_loss_euclidean_margin(self, y, margin=1.):
        return _loss_euclidean_margin(
            self.x1_dropout_layers[-1].output,
            self.x2_dropout_layers[-1].output, y, margin
            )

    def cos_same(self, y):
        """
        Return symbolic expression for the mean cosine distance of the same
        pairs alone.
        """
        cos_same_distances = T.switch(
            y, cos_distance(self.x1_layers[-1].output, self.x2_layers[-1].output), 0.
            )
        return T.sum(cos_same_distances) / T.sum(y)  # normalize by number of same pairs

    def cos_diff(self, y):
        """
        Return symbolic expression for the mean cosine distance of the
        different pairs alone.
        """
        cos_diff_distances = T.switch(
            y, 0., cos_distance(self.x1_layers[-1].output, self.x2_layers[-1].output)
            )
        return T.sum(cos_diff_distances) / T.sum(1 - y)  # normalize by number of different pairs

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        for layer in self.layers:
            layer.load(f)


#-----------------------------------------------------------------------------#
#                          SIAMESE TRIPLET CNN CLASS                          #
#-----------------------------------------------------------------------------#

class SiameseTripletCNN(object):
    """
    Siamese triplet convolutional neural network, allowing for hinge losses.

    An example use of this network is to train a metric on triplets A, B, X.
    Assume A-X is a same pair and B-X a different pair. Then a cost can be
    defined such that dist(A, X) > dist(B, X) by some margin. By convention the 
    same-pair is taken as `x1_layers` and `x2_layers`, while the different pair
    is taken as `x1_layers` and `x3_layers`.

    Attributes
    ----------
    x1_layers : list of ConvMaxPoolLayer and HiddenLayer
        Attributes are similar to `SiameseCNN`, except that now there are
        three tied networks, and so there is `x1_layers`, `x2_layers` and
        `x3_layers`, with corresponding additional layers when using dropout.
    """

    def __init__(self, rng, input_x1, input_x2, input_x3, input_shape,
            conv_layer_specs, hidden_layer_specs, srng=None,
            dropout_rates=None, activation=T.tanh):
        """
        Initialize symbolic parameters and expressions.

        Many of the parameters are identical to that of `cnn.build_cnn_layers`.
        Some of the other parameters are described below.

        Parameters
        ----------
        input_x1 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the first side of the Siamese network.
        input_x2 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the second side of the Siamese network, forming a
            same-pair with `input_x1`.
        input_x3 : symbolic matrix
            The matrix is reshaped according to `input_shape` and then treated
            as the input of the third side of the Siamese network, forming a
            different-pair with `input_x1`.
        """

        # Build common layers to which the Siamese layers are tied
        input = T.matrix("x")
        self.input = input
        layers = cnn.build_cnn_layers(
            rng, input, input_shape, conv_layer_specs, hidden_layer_specs,
            srng, dropout_rates
            )

        # Copy the shared variables
        shared_W = []
        shared_b = []
        # The isinstance deals with dropout (where `layers` is a tuple)
        for layer in (layers[1] if isinstance(layers, tuple) else layers):
            shared_W.append(layer.W)
            shared_b.append(layer.b)

        # Build the Siamese layers and tie their weights
        x1_layers = cnn.build_cnn_layers(
            rng, input_x1, input_shape, conv_layer_specs,
            hidden_layer_specs, srng, dropout_rates,
            init_W=shared_W, init_b=shared_b
            )
        x2_layers = cnn.build_cnn_layers(
            rng, input_x2, input_shape, conv_layer_specs,
            hidden_layer_specs, srng, dropout_rates,
            init_W=shared_W, init_b=shared_b
            )
        x3_layers = cnn.build_cnn_layers(
            rng, input_x3, input_shape, conv_layer_specs,
            hidden_layer_specs, srng, dropout_rates,
            init_W=shared_W, init_b=shared_b
            )

        if dropout_rates is not None:
            # All the layers so far is actually tuples which includes the dropout layers
            self.dropout_layers, self.layers = layers
            self.x1_dropout_layers, self.x1_layers = x1_layers
            self.x2_dropout_layers, self.x2_layers = x2_layers
            self.x3_dropout_layers, self.x3_layers = x3_layers
        else:
            self.layers = layers
            self.x1_layers = x1_layers
            self.x2_layers = x2_layers
            self.x3_layers = x3_layers

        # Model parameters
        self.parameters = []
        self.l1 = 0.
        self.l2 = 0.
        for layer in self.layers:
            self.parameters += layer.parameters
            self.l1 += abs(layer.W).sum()
            self.l2 += (layer.W**2).sum()

    def loss_hinge_cos(self, margin=0.5):
        return _loss_hinge_cos(
            self.x1_layers[-1].output,
            self.x2_layers[-1].output,
            self.x3_layers[-1].output,
            margin
            )

    def dropout_loss_hinge_cos(self, margin=0.5):
        return _loss_hinge_cos(
            self.x1_dropout_layers[-1].output,
            self.x2_dropout_layers[-1].output,
            self.x3_dropout_layers[-1].output,
            margin
            )

    def cos_same(self):
        """
        Return symbolic expression for the mean cosine distance of the same
        pairs alone.
        """
        return T.mean(cos_distance(self.x1_layers[-1].output, self.x2_layers[-1].output))

    def cos_diff(self):
        """
        Return symbolic expression for the mean cosine distance of the
        different pairs alone.
        """
        return T.mean(cos_distance(self.x1_layers[-1].output, self.x3_layers[-1].output))

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        for layer in self.layers:
            layer.save(f)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        for layer in self.layers:
            layer.load(f)


#-----------------------------------------------------------------------------#
#                            LOSS UTILITY FUNCTIONS                           #
#-----------------------------------------------------------------------------#

def cos_similarity(x1, x2):
    return (
        T.sum(x1 * x2, axis=-1) /
        (x1.norm(2, axis=-1) * x2.norm(2, axis=-1))
        )
cos_distance = lambda x1, x2: (1. - cos_similarity(x1, x2)) / 2.


def _loss_cos_cos2(x1, x2, y):
    cos_cos2 = T.switch(
        y, (1. - cos_similarity(x1, x2)) / 2., cos_similarity(x1, x2)**2
        )
    return T.mean(cos_cos2)


def _loss_cos_cos(x1, x2, y):
    cos_cos = T.switch(
        y, (1. - cos_similarity(x1, x2)) / 2., (cos_similarity(x1, x2) + 1.0) / 2.
        )
    return T.mean(cos_cos)


def _loss_cos_cos_margin(x1, x2, y, margin):
    loss_same = (1. - cos_similarity(x1, x2)) / 2.
    loss_diff = T.maximum(0., (cos_similarity(x1, x2) + 1.0) / 2. - margin)
    cos_cos = T.switch(
        y, loss_same, loss_diff
        )
    return T.mean(cos_cos)


def _loss_euclidean_margin(x1, x2, y, margin):
    loss_same = ((x1 - x2).norm(2, axis=-1))**2
    loss_diff = (T.maximum(0., margin - (x1 - x2).norm(2, axis=-1)))**2
    return T.mean(
        T.switch(y, loss_same, loss_diff)
        )


def _loss_hinge_cos(x1, x2, x3, margin):
    return T.mean(T.maximum(
        0.,
        margin + cos_distance(x1, x2) - cos_distance(x1, x3)
        ))


#-----------------------------------------------------------------------------#
#                                TEST FUNCTIONS                               #
#-----------------------------------------------------------------------------#

def np_loss_cos_cos2(x1, x2, y):
    assert x1.shape[0] == x2.shape[0] == y.shape[0]
    losses = []
    for i in xrange(x1.shape[0]):
        if y[i] == 1:
            # Data points are the same, use cosine distance
            loss = distance.cosine(x1[i], x2[i]) / 2.
            losses.append(loss)
        elif y[i] == 0:
            # Data points are different, use cosine similarity squared
            loss = (distance.cosine(x1[i], x2[i]) - 1)**2
            losses.append(loss)
        else:
            assert False
    return np.mean(losses)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    
    from cnn import np_cnn_layers_output
    from theano.tensor.shared_randomstreams import RandomStreams
    import itertools
    import numpy.testing as npt
    import theano_utils


    # Test `SiameseCNN`

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
    activation = theano_utils.relu
    model = SiameseCNN(
            rng, input_x1, input_x2, input_shape,
            conv_layer_specs, hidden_layer_specs,
            srng, dropout_rates=dropout_rates, 
            activation=activation
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
        hidden_layers_W, hidden_layers_b, activation=theano_utils.np_relu
        )
    np_x2_layers_output = np_cnn_layers_output(
        X[x2_indices], conv_layers_W, conv_layers_b, conv_layers_pool_shape,
        hidden_layers_W, hidden_layers_b, activation=theano_utils.np_relu
        )

    numpy_loss = np_loss_cos_cos2(np_x1_layers_output, np_x2_layers_output, Y)
    print "Numpy loss:", numpy_loss

    npt.assert_almost_equal(numpy_loss, theano_loss)


if __name__ == "__main__":
    main()

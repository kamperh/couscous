"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import copy
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T

import logistic
import theano_utils


#-----------------------------------------------------------------------------#
#                              HIDDEN LAYER CLASS                             #
#-----------------------------------------------------------------------------#

class HiddenLayer(object):
    """
    Fully connected hidden layer used in an MLP.

    Attributes
    ----------
    W : shared matrix of shape (d_in, d_out)
        Weights from the d_in input units to the d_out output units.
    b : shared vector of size d_out
        The biases for each of the d_out output units.
    output : symbolic expression
        The output of the hidden layer given the input and current parameters.
    """

    def __init__(self, rng, input, d_in, d_out, W=None, b=None,
            activation=T.tanh):
        """
        Initialize symbolic parameters and expressions.

        If `W` is not given, it is initialized between the following values
        according to the activation function:
        - None and tanh: sqrt(-6./(d_in+d_out)) and sqrt(6./(d_in+d_out))
        - sigmoid: 4 times that of tanh

        Parameters
        ----------
        rng : numpy.random.RandomState
            Random number generator used for weight initialization.
        input : symbolic matrix of shape (n_data, d_in)
            Symbolic matrix describing the input (i.e. one minibatch), with
            each row representing a data instance.
        d_in : int
            Number of input units.
        d_out : int
            Number of output units.
        activation : operation
            Can be theano.tensor.tanh, theano.tensor.nnet.sigmoid or None
            (which specifies linear activation).
        """

        # Initialize weights
        if W is None:
            # W_values = np.asarray(rng.uniform(
            #     low=-np.sqrt(6. / (d_in + d_out)),
            #     high=np.sqrt(6. / (d_in + d_out)),
            #     size=(d_in, d_out)
            #     ), dtype=theano.config.floatX)
            if activation == T.tanh:
                W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (d_in + d_out)),
                    high=np.sqrt(6. / (d_in + d_out)),
                    size=(d_in, d_out)
                    ), dtype=theano.config.floatX)
            elif activation == theano.tensor.nnet.sigmoid:
                W_values = 4. * np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (d_in + d_out)),
                    high=np.sqrt(6. / (d_in + d_out)),
                    size=(d_in, d_out)
                    ), dtype=theano.config.floatX)
            elif activation == theano_utils.relu:
                W_values = np.asarray(0.01*rng.randn(d_in, d_out), dtype=theano.config.floatX)
                # W_values *= 4
            elif activation is None:
                # Linear activation
                W_values = np.asarray(0.01*rng.randn(d_in, d_out), dtype=theano.config.floatX)
            else:
                assert False, "Invalid activation: " + str(activation)
            W = theano.shared(value=W_values, name="W", borrow=True)

        # Initialize biases
        if b is None:
            b_values = np.zeros((d_out,), dtype=theano.config.floatX)
            # if activation == theano_utils.relu:
            #     b_values = 0.01*np.ones((d_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b", borrow=True)

        self.W = W
        self.b = b

        # Symbolic expression for hidden layer output
        linear_output = T.dot(input, self.W) + self.b
        self.output = (linear_output if activation is None else activation(linear_output))

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
#                          DROPOUT HIDDEN LAYER CLASS                         #
#-----------------------------------------------------------------------------#

class DropoutHiddenLayer(HiddenLayer):
    """
    Dropout is applied to this hidden layer's output.

    Loosely based on the following:
    - https://github.com/mdenil/dropout/blob/master/mlp.py
    - https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py
    """
    def __init__(self, rng, srng, dropout_rate, input, d_in, d_out,
            W=None, b=None, activation=T.tanh):
        """
        Apart from the `srng` and `dropout_rate`, the parameters are
        identical to those of `HiddenLayer`.
        """
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, d_in=d_in, d_out=d_out, W=W, b=b,
            activation=activation
            )
        self.output = theano_utils.apply_dropout(srng, self.output, p=dropout_rate)


#-----------------------------------------------------------------------------#
#                                  MLP CLASS                                  #
#-----------------------------------------------------------------------------#

class MLP(object):
    """
    Multilayer perceptron.

    Attributes
    ----------
    layers : list of HiddenLayer and LogisticRegression
        The hidden layers of the MLP, as well as the final logistic regression
        layer.
    y_pred : symbolic expression
        Given by `y_pred` from the logistic regression layer.
    negative_log_likelihood : symbolic expression
        Given by `negative_log_likelihood` from the logistic regression layer.
    errors : symbolic expression
        Given by `errors` from the logistic regression layer.
    """

    def __init__(self, rng, input, d_in, d_out, hidden_layer_specs, srng=None,
            dropout_rates=None):
        """
        Initialize symbolic parameters and expressions.

        Most of the parameters are identical to that of `build_mlp_layers`,
        which is used to build all the layers of the MLP, with a logistic
        regression layer added on top.

        Parameters
        ----------
        rng : numpy.random.RandomState
            Random number generator used for weight initialization.
        d_in : int
            Dimensionality of input features.
        d_out : int
            Number of output classes.
        hidden_layer_sizes : list of int
            The size of each of the hidden layers.
        hidden_layer_activation : operatoin
            See `activation` parameter of `HiddenLayer`.
        """

        self.input = input
        assert len(hidden_layer_specs) > 0

        if dropout_rates is not None:
            self.dropout_layers, self.layers = build_mlp_layers(
                rng, input, d_in, hidden_layer_specs, srng, dropout_rates
                )
        else:
            self.layers = build_mlp_layers(rng, input, d_in, hidden_layer_specs)

        # self.layers = []
        # n_layers = len(hidden_layer_sizes)
        # assert n_layers > 0

        # # Build hidden layers
        # for i_layer in xrange(n_layers):

        #     if i_layer == 0:
        #         cur_d_in = d_in
        #         cur_input = input
        #     else:
        #         cur_d_in = hidden_layer_sizes[i_layer - 1]
        #         cur_input = self.layers[i_layer - 1].output

        #     hidden_layer = HiddenLayer(
        #         rng=rng,
        #         input=cur_input,
        #         d_in=cur_d_in,
        #         d_out=hidden_layer_sizes[i_layer],
        #         activation=hidden_layer_activation
        #         )

        #     self.layers.append(hidden_layer)

        #     # self.l1 += abs(self.layers[-1].W).sum()
        #     # self.l2 += (self.layers[-1].W ** 2).sum()
        #     # self.parameters.extend(hidden_layer.parameters)

        # Logistic regression class prediciton layer
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
        self.l1 = 0
        self.l2 = 0
        for layer in self.layers:
            self.parameters += layer.parameters
            self.l1 += abs(layer.W).sum()
            self.l2 += (layer.W**2).sum()

        # Symbolic expressions of log likelihood loss and prediction error
        self.y_pred = self.layers[-1].y_pred
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

def build_mlp_layers(rng, input, d_in, hidden_layer_specs, srng=None,
        dropout_rates=None, init_W=None, init_b=None):
    """
    Return the fully-connected hidden layers of an MLP.

    The layer specifications can look something like this:

        hidden_layer_specs = [
        {"units": 500, "activation": theano_utils.relu},
        {"units": 500, "activation": "sigmoid"}
        ]

    Parameters
    ----------
    input : symbolic matrix of shape (n_data, d_in)
        Input to the first layer of the MLP.
    d_in : int
        Number of input units.
    hidden_layer_specs : list of dict
        Specifications for the fully-connected hidden layers.
    dropout_rates : list of float
        The dropout rates for each of the layers; if not provided, dropout is
        not performed.
    init_W : list of shared tensors
        If provided, these weights are used for layer initialization. The
        weights should be given in the same order that the layers are created.
        This is useful for tying weights.
    init_b : list of shared vectors
        If provided, these biases are used for layer initialization. The order
        should be the same as that of `init_W`.
    """

    assert dropout_rates is None or len(dropout_rates) == len(hidden_layer_specs)

    if init_W is not None:
        assert init_b is not None

        # We are going to pop parameters, so make copies
        init_W = init_W[:]
        init_b = init_b[:]

    hidden_layer_specs = copy.deepcopy(hidden_layer_specs)
    for layer_spec in hidden_layer_specs:
        activation_str_to_op(layer_spec)

    layers = []
    if dropout_rates is not None:
        dropout_layers = []

    for i_layer in xrange(len(hidden_layer_specs)):

        if i_layer == 0:
            cur_d_in = d_in
            cur_input = input
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
        layer = HiddenLayer(
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
                cur_dropout_input = input
            else:
                cur_dropout_input = dropout_layers[-1].output
            dropout_rate = dropout_rates[i_layer]
            dropout_layer = DropoutHiddenLayer(
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


def activation_str_to_op(layer_spec):
    if "activation" in layer_spec:
        if layer_spec["activation"] == "tanh":
            layer_spec["activation"] = T.tanh
        elif layer_spec["activation"] == "sigmoid":
            layer_spec["activation"]  = T.nnet.sigmoid
        elif layer_spec["activation"] == "relu":
            layer_spec["activation"] = theano_utils.relu
        elif layer_spec["activation"] == "linear":
            layer_spec["activation"] = None


#-----------------------------------------------------------------------------#
#                                TEST FUNCTIONS                               #
#-----------------------------------------------------------------------------#

def np_mlp_negative_log_likelihood(X, y, hidden_layers_W, hidden_layers_b,
        logistic_regression_W, logistic_regression_b,
        hidden_layer_activation=np.tanh):
    """
    Numpy function for the negative mean log likelihood of the data `X` with
    labels `y` of an MLP with the given parameters, assuming tanh activation.
    """

    from logistic import np_negative_log_likelihood

    # Calculate hidden layer outputs
    for i_layer in range(len(hidden_layers_W)):
        cur_W = hidden_layers_W[i_layer]
        cur_b = hidden_layers_b[i_layer]

        if i_layer == 0:
            cur_input = X
        else:
            cur_input = output

        # Calculate activation
        output = hidden_layer_activation(np.dot(cur_input, cur_W) + cur_b)

        # # Add regularization
        # l1 += abs(cur_W).sum()
        # l2 += (cur_W**2).sum()

    # Calculate logistic regression cost
    return np_negative_log_likelihood(output, y, logistic_regression_W, logistic_regression_b)


def np_mlp_layers_output(X, hidden_layers_W, hidden_layers_b,
        hidden_layers_activation):
    """
    Numpy function that returns the output of the MLP layers (specified by the
    given parameters) when presented with input data `X`.
    """
    for i_layer in range(len(hidden_layers_W)):
        cur_W = hidden_layers_W[i_layer]
        cur_b = hidden_layers_b[i_layer]
        cur_activation = hidden_layers_activation[i_layer]

        if i_layer == 0:
            cur_input = X
        else:
            cur_input = output

        # Calculate activation
        output = cur_activation(np.dot(cur_input, cur_W) + cur_b)

    return output
    

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import theano_utils
    
    # Setup model
    rng = np.random.RandomState(42)
    d_in = 20
    d_out = 15
    l1_weight=0.01
    l2_weight=0.0001
    hidden_layer_sizes = [3, 4]
    hidden_layer_activation = theano_utils.relu
    x = T.matrix("x")       # input: feature vectors
    y = T.ivector("y")      # output: int values for each class
    model = MLP(
        rng, input=x, d_in=d_in, d_out=d_out,
        hidden_layer_specs=[
            {"units": units, "activation": hidden_layer_activation} for units in hidden_layer_sizes
            ]
        )
    cost = model.negative_log_likelihood(y) + l1_weight * model.l1 + l2_weight * model.l2

    # Generate non-zero weights and biases and assign to logistic regression
    W_test = rng.randn(hidden_layer_sizes[-1], d_out)
    b_test = rng.randn(d_out)
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
    N = 10
    X = rng.randn(N, d_in)
    y = rng.randint(d_out, size=N)
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


if __name__ == "__main__":
    main()

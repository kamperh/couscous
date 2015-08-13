"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T


class LogisticRegression(object):
    """
    Multi-class logistic regression (i.e. softmax).

    Attributes
    ----------
    W : shared matrix of shape (d_in, d_out)
        Each of the d_out columns is the weight vector for one of the d_out
        classes, with d_in the dimensionality of the input units.
    b : shared vector of size d_out
        The biases for each of the d_out classes.
    linear_output : symbolic expression
        This is the linear operation before the softmax, which is used for
        calculating `p_y_given_x` below.
    p_y_given_x : symbolic expression
        Symbolic expression for the class probability, given as a matrix with
        the data instances indexed across the rows and the classes indexed
        across the columns.
    y_pred : symbolic expression
        Symbolic expression for class prediction.
    """

    def __init__(self, input, d_in, d_out, W=None, b=None):
        """
        Initialize symbolic parameters and prediction expressions.

        Parameters
        ----------
        input : symbolic matrix of shape (n_data, d_in)
            Symbolic matrix describing the input (i.e. one minibatch), with
            each row representing a data instance.
        d_in : int
            Number of input units.
        d_out : int
            Number of output classes.
        """

        # Initialize the weights and biases
        if W is None:
            self.W = theano.shared(
                value=np.zeros((d_in, d_out), dtype=theano.config.floatX),
                name="W",
                borrow=True
                )
        else:
            self.W = W
        if b is None:
            self.b = theano.shared(
                value=np.zeros((d_out,), dtype=theano.config.floatX),
                name="b",
                borrow=True
                )
        else:
            self.b = b

        self.linear_output = T.dot(input, self.W) + self.b

        # Symbolic expression for the class probability; this is a matrix where
        # each row corresponds to a data instance and each column corresponds
        # to a class
        self.p_y_given_x = T.nnet.softmax(self.linear_output)

        # Symbolic expression for class prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Model parameters
        self.parameters = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        Return a symbolic expression for the negative mean log likelihood of
        data under the current model.

        Note that implicitly this is calculated for the data given as symbolic
        variable `input` passed during initialization, i.e. the current
        minibatch.

        Parameters
        ----------
        y : symbolic vector of size n_data
            Symbolic vector that gives the correct label for each data
            instance (implicitly, each data instance in `input` passed in
            initialization).
        """
        # The returned expression is built up as follows:
        # - y.shape[0]: Number of data instances.
        # - T.arange(y.shape[0]): A list [0, 1, ..., N-1] which indexes every
        #   data instance.
        # - T.log(self.p_y_given_x): Matrix of log probabilities for each data
        #   item (rows) belonging to each of the classes (columns).
        # - [T.arange(y.shape[0]), y]: Indexes the log probabilities of the 
        #   the correct class labels.
        # - T.mean(...): The mean log likelihood across the minibatch; the mean
        #   is used since the learning rate is then less dependent on the batch
        #   size.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        Return a symbolic expression for the number of errors under the current
        model.

        Parameters
        ----------
        y : symbolic vector of size n_data
            Same format as `y` parameter of `negative_log_likelihood`.
        """
        assert y.ndim == self.y_pred.ndim
        return T.mean(T.neq(self.y_pred, y))

    def save(self, f):
        """Pickle the model parameters to opened file `f`."""
        pickle.dump(self.W.get_value(borrow=True), f, -1)
        pickle.dump(self.b.get_value(borrow=True), f, -1)

    def load(self, f):
        """Load the model parameters from the opened pickle file `f`."""
        self.W.set_value(pickle.load(f), borrow=True)
        self.b.set_value(pickle.load(f), borrow=True)


def np_negative_log_likelihood(X, y, W, b):
    """
    Numpy function for the negative mean log likelihood of the data `X` with
    labels `y` of a logistic regression model with weights `W` and biases `b`.
    """

    # Every row is the class probability of that data point
    numerator = np.exp(np.dot(X, W) + b)
    p_y_given_x = numerator / np.sum(numerator, axis=1)[:, None]
    
    log_likelihoods = np.log(p_y_given_x[range(y.shape[0]), y])

    negative_log_likelihood = -np.mean(log_likelihoods)

    return negative_log_likelihood


def main():
    
    # Setup model
    d_in = 20
    d_out = 15
    x = T.matrix("x")       # input: feature vectors
    y = T.ivector("y")      # output: int values for each class
    model = LogisticRegression(input=x, d_in=d_in, d_out=d_out)
    cost = model.negative_log_likelihood(y)

    # Generate non-zero weights and biases and assign to model
    W_test = np.random.randn(d_in, d_out)
    b_test = np.random.randn(d_out)
    model.W.set_value(W_test)
    model.b.set_value(b_test)

    # Compile a test function
    test_model = theano.function(inputs=[x, y], outputs=cost)

    # Generate random data
    N = 2
    X = np.random.randn(N, d_in)
    y = np.random.randint(d_out, size=N)
    y = y.astype("int32")

    # Calculate Numpy cost
    np_cost = np_negative_log_likelihood(X, y, W_test, b_test)
    print "Numpy cost:", np_cost

    # Calculate Theano cost
    theano_cost = test_model(X, y)
    print "Theano cost:", theano_cost


if __name__ == "__main__":
    main()

"""
Training functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015
"""

from theano.compat.python2x import OrderedDict
from datetime import datetime
import cPickle as pickle
import logging
import numpy as np
import timeit
import theano
import theano.tensor as T

from data_io import smart_open

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                           LEARNING RULE FUNCTIONS                           #
#-----------------------------------------------------------------------------#

def learning_rule_momentum(parameters, gradients, learning_rate, momentum):
    """
    Return a list of the momentum updates.

    Based loosely on:
    - The Pylearn2 function:
      `pylearn2.training_algorithms.learning_rule.Momentum.get_updates`
    - G. Hinton, "A practical guide to training restricted Boltzmann machines".

    Parameters
    ----------
    parameters : list of shared variables
        Parameters to update.
    gradients : list symbolic expression
        List of gradients for each of the parameters.
    learning_rate : float 
        Gradient descent learning rate.
    momentum : float
        Momentum parameter: should be between 0 and 1.
    """
    assert momentum < 1 and momentum >= 0

    updates = OrderedDict()

    for parameter, grad in zip(parameters, gradients):

        # Initialize the shared variable for the step for this parameter
        delta_parameter = theano.shared(parameter.get_value(borrow=True)*0.)

        # Update the step
        updates[delta_parameter] = momentum*delta_parameter - learning_rate*grad

        # Update the parameters using the step
        updates[parameter]  = parameter + delta_parameter

    return updates


def learning_rule_adagrad(parameters, gradients, learning_rate=1.0,
        max_scaling=1e5):
    """
    Return an OrderedDict of the AdaGrad updates.

    Based loosely on:
    - The Pylearn2 function:
      `pylearn2.training_algorithms.learning_rule.AdaGrad.get_updates`
    - J. Duchi, E. Hazan, and Y. Singer, "Adaptive subgradient methods for
      online learning and stochastic optimization," J. Mach. Learn. Res., vol.
      12, pp. 2121-2159, 2011.

    Parameters
    ----------
    parameters : list of shared variables
        Parameters to update.
    gradients : list symbolic expression
        List of gradients for each of the parameters.
    learning_rate : float
        The global learning rate.
    """

    updates = OrderedDict()

    for parameter, gradient in zip(parameters, gradients):

        sum_square_gradient = theano.shared(parameter.get_value(borrow=True)*0.)

        if parameter.name is not None:
            sum_square_gradient.name = "sum_square_gradient_" + parameter.name

        # Accumulate gradient
        new_sum_squared_gradient = sum_square_gradient + T.sqr(gradient)

        # Compute the update step
        epsilon = learning_rate
        scale = T.maximum(1./max_scaling, T.sqrt(new_sum_squared_gradient))
        delta_parameter = -epsilon/scale*gradient

        # Update the parameter using the step
        updates[sum_square_gradient] = new_sum_squared_gradient
        updates[parameter] = parameter + delta_parameter

    return updates


def learning_rule_adadelta(parameters, gradients, rho=0.9, epsilon=1e-6):
    """
    Return an OrderedDict of the AdaDelta updates.

    Based loosely on:
    - The Pylearn2 function:
      `pylearn2.training_algorithms.learning_rule.AdaDelta.get_updates`
    - M.D. Zeiler, "ADADELTA: An adaptive learning rate method," arXiv preprint
      arXiv:1212.5701, 2012.

    Parameters
    ----------
    parameters : list of shared variables
        Parameters to update.
    gradients : list symbolic expression
        List of gradients for each of the parameters.
    rho : float
        Decay rate parameter.
    epsilon : float
        Precision of updates.
    """

    updates = OrderedDict()

    for parameter, gradient in zip(parameters, gradients):

        mean_square_gradient = theano.shared(parameter.get_value(borrow=True)*0.)
        mean_square_delta_parameter = theano.shared(parameter.get_value(borrow=True)*0.)

        if parameter.name is not None:
            mean_square_gradient.name = "mean_square_gradient_" + parameter.name
            mean_square_delta_parameter.name = "mean_square_delta_parameter_" + parameter.name

        # Accumulate gradient
        new_mean_squared_gradient = (
            rho * mean_square_gradient + (1 - rho) * T.sqr(gradient)
            )

        # Compute the update step
        rms_delta_parameter_t_min_1 = T.sqrt(mean_square_delta_parameter + epsilon)
        rms_gradient_t = T.sqrt(new_mean_squared_gradient + epsilon)
        delta_parameter = - rms_delta_parameter_t_min_1 / rms_gradient_t * gradient

        # Accumulate updets
        new_mean_square_delta_parameter = (
            rho*mean_square_delta_parameter + (1 - rho) * T.sqr(delta_parameter)
            )

        # Update the parameter using the step
        updates[mean_square_gradient] = new_mean_squared_gradient
        updates[mean_square_delta_parameter] = new_mean_square_delta_parameter
        updates[parameter] = parameter + delta_parameter

    return updates



#-----------------------------------------------------------------------------#
#                              TRAINING FUNCTIONS                             #
#-----------------------------------------------------------------------------#

def train_fixed_epochs(n_epochs, train_model, train_batch_iterator,
        test_model=None, test_batch_iterator=None, save_model_func=None,
        save_model_fn=None, record_dict_fn=None):
    """
    Train for a fixed number of epochs.

    Parameters
    ----------
    train_model : Theano function
        Should take input from `train_batch_iterator` and output the training
        loss. The function can provide more than one output, which is averaged.
        This is useful for example to output both negative log likelihood (the
        model loss) and zero-one loss (the number of errors).
    train_batch_iterator : generator
        Provides the training batches.
    save_model_func : function
        If provided, this function is used to the save the model to the file
        `save_model_fn` every time a new validation best model is found.
    save_model_fn : str
        The file to which the best model is written.
    record_dict_fn : str
        If provided, the current `record_dict` is saved to this file at the end
        of every epoch.

    Return
    ------
    record_dict : dict
        The dict key describes the statistic being tracked, while the dict
        value is a list of (epoch, statistic) tuples giving the statistic-value
        at a particular epoch.
    """

    record_dict = {}
    record_dict["train_loss"] = []          # each element is (epoch, loss)
    if test_model is not None:
        record_dict["test_loss"] = []       # testing is not necessarily performed every epoch
    record_dict["epoch_time"] = []

    logger.info(datetime.now())

    # Training epochs
    i_epoch_best = 0
    test_loss = np.inf
    for i_epoch in xrange(n_epochs):

        # Loop over training batches
        # train_losses = []
        start_time = timeit.default_timer()
        train_losses = [train_model(*batch) for batch in train_batch_iterator]
        # for i_batch in xrange(n_train_batches):
        # for batch in train_batch_iterator()
            # Calculate training loss for this batch and update parameters
            # train_losses.append(train_model(*batch))

        # Test model
        if test_model is not None:
            test_losses = [test_model(*batch) for batch in test_batch_iterator]
            test_loss = np.mean(test_losses, axis=0)
            logger.info("    Test loss: " + str(test_loss))
            record_dict["test_loss"].append((i_epoch, test_loss))

        # Write this model
        if save_model_func is not None:
            f = smart_open(save_model_fn, "wb")
            save_model_func(f)
            f.close()

        # Training statistics for this epoch
        end_time = timeit.default_timer()
        train_loss = np.mean(train_losses, axis=0)
        epoch_time = end_time - start_time
        # logger.info("Training loss: " + str(train_loss)  # + ", " + 
            # )
        logger.info("Time: %f" % (epoch_time) + " sec, " + 
            "training loss: " + str(train_loss)  # + ", " + 
            )
        record_dict["epoch_time"].append((i_epoch, epoch_time))
        record_dict["train_loss"].append((i_epoch, train_loss))

        if record_dict_fn is not None:
            f = smart_open(record_dict_fn, "wb")
            pickle.dump(record_dict, f, -1)
            f.close()

    total_time = np.sum([i[1] for i in record_dict["epoch_time"]])
    logger.info("Training complete: %f min" % (total_time / 60.))
    if test_model is not None:
        logger.info("Test loss: " + str(test_loss))
    if save_model_func is not None:
        logger.info("Model saved: " + save_model_fn)
    if record_dict_fn is not None:
        logger.info("Saved record: " + record_dict_fn)

    logger.info(datetime.now())

    return record_dict


def train_fixed_epochs_with_validation(n_epochs, train_model,
        train_batch_iterator, validate_model, validate_batch_iterator,
        test_model=None, test_batch_iterator=None, save_model_func=None,
        save_model_fn=None, record_dict_fn=None):
    """
    Train for a fixed number of epochs, using validation to decide which model
    to save.

    Parameters
    ----------
    train_model : Theano function
        Should take input from `train_batch_iterator` and output the training
        loss. The function can provide more than one output, which is averaged.
        This is useful for example to output both negative log likelihood (the
        model loss) and zero-one loss (the number of errors).
    train_batch_iterator : generator
        Provides the training batches.
    validate_model : Theano function
        Should take input from `validate_batch_iterator` and output the
        validation loss. The function can provide more than one output (which
        would be averaged), but for the validation only the first output will
        be used (except if `validate_extrinsic` is provided).
    validate_extrinsic : function
        Extrinsic evaluation can be performed using this function. If provided,
        validation is performed on the output of this function instead of using
        the output from `validate_model`.
    save_model_func : function
        If provided, this function is used to the save the model to the file
        `save_model_fn` every time a new validation best model is found.
    save_model_fn : str
        The file to which the best model is written.
    record_dict_fn : str
        If provided, the current `record_dict` is saved to this file at the end
        of every epoch.

    Return
    ------
    record_dict : dict
        The dict key describes the statistic being tracked, while the dict
        value is a list of (epoch, statistic) tuples giving the statistic-value
        at a particular epoch.
    """

    record_dict = {}
    record_dict["train_loss"] = []          # each element is (epoch, loss)
    record_dict["validation_loss"] = []     # validation is not necessarily performed every epoch
    if test_model is not None:
        record_dict["test_loss"] = []       # and neither is testing
    # if validate_extrinsic is not None:
    #     record_dict["validation_extrinsic"] = []
    record_dict["epoch_time"] = []

    logger.info(datetime.now())

    # Training epochs
    best_validation_loss0 = np.inf
    test_loss = np.inf
    i_epoch_best = 0
    for i_epoch in xrange(n_epochs):

        # Loop over training batches
        # train_losses = []
        start_time = timeit.default_timer()
        train_losses = [train_model(*batch) for batch in train_batch_iterator]
        # for i_batch in xrange(n_train_batches):
        # for batch in train_batch_iterator()
            # Calculate training loss for this batch and update parameters
            # train_losses.append(train_model(*batch))

        # Validate the model
        validation_losses = [validate_model(*batch) for batch in validate_batch_iterator]
        validation_loss = np.mean(validation_losses, axis=0)
        logger.info("Epoch " + str(i_epoch + 1) + ": "
            "validation loss: " + str(validation_loss)
            )
        record_dict["validation_loss"].append((i_epoch, validation_loss))
        
        # print math.isnan(validation_loss)
        if hasattr(validation_loss, "__len__"):
            validation_loss0 = validation_loss[0]
        else:
            validation_loss0 = validation_loss

        # If this is the best model, test and save
        if validation_loss0 < best_validation_loss0:

            best_validation_loss0 = validation_loss0
            i_epoch_best = i_epoch

            # Test model
            if test_model is not None:
                test_losses = [test_model(*batch) for batch in test_batch_iterator]
                test_loss = np.mean(test_losses, axis=0)
                logger.info("    Test loss: " + str(test_loss))
                record_dict["test_loss"].append((i_epoch, test_loss))

            # Write the best model
            if save_model_func is not None:
                f = smart_open(save_model_fn, "wb")
                save_model_func(f)
                f.close()

        # Training statistics for this epoch
        end_time = timeit.default_timer()
        train_loss = np.mean(train_losses, axis=0)
        epoch_time = end_time - start_time
        # logger.info("Training loss: " + str(train_loss)  # + ", " + 
            # )
        logger.info("Time: %f" % (epoch_time) + " sec, " + 
            "training loss: " + str(train_loss)  # + ", " + 
            )
        record_dict["epoch_time"].append((i_epoch, epoch_time))
        record_dict["train_loss"].append((i_epoch, train_loss))

        if record_dict_fn is not None:
            f = smart_open(record_dict_fn, "wb")
            pickle.dump(record_dict, f, -1)
            f.close()

    total_time = np.sum([i[1] for i in record_dict["epoch_time"]])
    logger.info("Training complete: %f min" % (total_time / 60.))
    logger.info(
        "Best validation epoch: " + str(i_epoch_best + 1) + ", "
        "best validation loss: " + str(best_validation_loss0)
        )
    if test_model is not None:
        logger.info("Test loss: " + str(test_loss))
    if save_model_func is not None:
        logger.info("Best validation model saved: " + save_model_fn)
    if record_dict_fn is not None:
        logger.info("Saved record: " + record_dict_fn)

    logger.info(datetime.now())

    return record_dict


def train_early_stopping(n_train_batches, n_validation_batches, train_model,
        validate_model, test_model=None, n_test_batches=None,
        n_max_epochs=1000, n_batches_validation_frequency=None,
        n_patience=5000, patience_increase_factor=2,
        improvement_threshold=0.995, save_model_func=None, save_model_fn=None,
        record_dict_fn=None, learning_rate_update=None):
    """
    Train model using early stopping, using the provided training function.

    Parameters
    ----------
    n_train_batches : int
        Total number of training batches.
    n_validation_batches : int
        Total number of validation batches.
    train_model : Theano function
        Should take as input a batch index and output the training loss and
        error (e.g. negative log likelihood and zero-one loss).
    validate_model : Theano function
        Should take as input a batch index and output the validation loss and
        error.
    test_model : Theano function
        Should take as input a batch index and output the test loss and error.
        If not provided, testing is not performed over the training iterations.
    n_test_batches : int
        Total number of test batches.
    n_batches_validation_frequency : int
        Number of batches between calculating the validation error; if not
        provided, is set to min(n_train_batches, n_patience / 2) which means
        that at a minimum validation will be performed every epoch (i.e. every
        time after seeing `n_train_batches` batches).
    n_patience : int
        Number of minibatches to consider at a minimum before completing
        training.
    patience_increase_factor : int
        When a new validation minimum is found, the number of seen minibatches
        are multiplied by this factor to give the new minimum number of
        minibatches before stopping.
    improvement_threshold : float
        The minimum relative improvement in validation error to be warrant an
        increase in `n_patience` by `patience_increase_factor`.
    save_model_func : function
        If provided, this function is used to the save the model to the file
        `save_model_fn` every time a new validation best model is found.
    save_model_fn : str
        The file to which the current model is written.
    record_dict_fn : str
        If provided, the current `record_dict` is saved to this file at the end
        of every epoch.
    learning_rate_update : Theano function
        If provided, this function is called (without any parameters) at the
        beginning of every epoch to update the learning rate.

    Return
    ------
    record_dict : dict
        The dict key describes the statistic being tract, while the dict value
        is a list of (epoch, statistic) tuples giving the statistic-value at a
        particular epoch.
    """

    assert (save_model_func is None) or (save_model_fn is not None)
    assert (test_model is None) or (n_test_batches is not None)

    # Set default if not provided
    if n_batches_validation_frequency is None:
        n_batches_validation_frequency = min(n_train_batches, n_patience / 2)

    record_dict = {}
    record_dict["train_loss"] = []          # each element is (epoch, loss)
    record_dict["train_error"] = []
    record_dict["validation_loss"] = []     # validation is not necessarily performed every epoch
    record_dict["validation_error"] = []
    if test_model is not None:
        record_dict["test_loss"] = []       # and neither is testing
        record_dict["test_error"] = []
    record_dict["epoch_time"] = []

    # Training epochs
    i_epoch = 0
    done_looping = False
    best_validation_error = np.inf
    n_batches_best = 0
    i_epoch_best = 0
    while (i_epoch < n_max_epochs) and (not done_looping):

        train_losses = []
        train_errors = []
        start_time = timeit.default_timer()

        if learning_rate_update is not None:
            learning_rate = learning_rate_update(i_epoch)

        # Minibatches
        for i_batch in xrange(n_train_batches):

            # Calculate cost for this minibatch, updating the parameters
            minibatch_train_loss, minibatch_train_errors = train_model(i_batch)
            train_errors.append(minibatch_train_errors)
            train_losses.append(minibatch_train_loss)

            # print train_losses
            # print i_batch, train_model(i_batch)
            # break

            n_seen_batches = i_epoch * n_train_batches + i_batch

            # Use n_seen_batches + 1 to avoid checking very first batch
            if (n_seen_batches + 1) % n_batches_validation_frequency == 0:

                # Validate model
                validation_losses_errors = [validate_model(i) for i in xrange(n_validation_batches)]
                validation_loss = np.mean([i[0] for i in validation_losses_errors])
                validation_error = np.mean([i[1] for i in validation_losses_errors])

                logger.info(
                    "Validation: epoch %i, minibatch %i/%i, loss %f, error %.2f%%" %
                    (i_epoch + 1, i_batch + 1, n_train_batches, validation_loss, validation_error * 100.)
                    )
                record_dict["validation_loss"].append((i_epoch, validation_loss))
                record_dict["validation_error"].append((i_epoch, validation_error))

                # Check validation to see if we have new best model
                if validation_error < best_validation_error:
                    if validation_error < best_validation_error * improvement_threshold:
                        n_patience = max(n_patience, n_seen_batches * patience_increase_factor)
                    best_validation_error = validation_error
                    n_batches_best = n_seen_batches
                    i_epoch_best = i_epoch

                    if test_model is not None:
                        # test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_losses_errors = [test_model(i) for i in xrange(n_test_batches)]
                        test_loss = np.mean([i[0] for i in test_losses_errors])
                        test_error = np.mean([i[1] for i in test_losses_errors])

                        logger.info("\tTest: loss %f, error %.2f%%" % (test_loss, test_error * 100.))
                        # logger.info(
                        #     "Test: epoch %i, minibatch %i/%i, error %f%%" %
                        #     (i_epoch + 1, i_batch + 1, n_train_batches, test_loss * 100)
                        #     )
                        record_dict["test_loss"].append((i_epoch, test_loss))
                        record_dict["test_error"].append((i_epoch, test_error))

                    # Write the best model
                    if save_model_func is not None:
                        f = smart_open(save_model_fn, "wb")
                        save_model_func(f)
                        f.close()

            # break

            # Check if training is done
            if n_patience <= n_seen_batches:
                done_looping = True
                break

        end_time = timeit.default_timer()
        epoch_time = end_time - start_time
        record_dict["epoch_time"].append((i_epoch, epoch_time))

        # print train_losses
        # print train_errors
        cur_train_loss = np.mean(train_losses)
        cur_train_error = np.mean(train_errors)
        if learning_rate_update is not None:
            logger.info(
                "Train: lr %f, epoch %i, %f sec/epoch, loss %f, error %.2f%%" % (
                    learning_rate, i_epoch + 1,
                    epoch_time, cur_train_loss, cur_train_error*100.
                    )
                )
        else:
            logger.info(
                "Train: epoch %i, %f sec/epoch, loss %f, error %.2f%%" % (
                    i_epoch + 1,
                    epoch_time, cur_train_loss, cur_train_error*100.
                    )
                )
        record_dict["train_loss"].append((i_epoch, cur_train_loss))
        record_dict["train_error"].append((i_epoch, cur_train_error))

        if record_dict_fn is not None:
            f = smart_open(record_dict_fn, "wb")
            pickle.dump(record_dict, f, -1)
            f.close()

        i_epoch += 1

    total_time = np.sum([i[1] for i in record_dict["epoch_time"]])
    logger.info(
        "Training complete: %d epochs, %f sec/epoch, total time %f min" %
        ( i_epoch, 1. * total_time / i_epoch, total_time / 60. )
        )
    logger.info(
        "Best validation: after seeing %d minibatches in epoch %d, error %.2f%%" %
        (n_batches_best, i_epoch_best + 1, best_validation_error * 100.)
        )
    if test_model is not None:
        logger.info("Test error: %.2f%%" % (test_error * 100.))
    if save_model_func is not None:
        logger.info("Best validation model saved: " + save_model_fn)
    if record_dict_fn is not None:
        logger.info("Saved record: " + record_dict_fn)

    return record_dict

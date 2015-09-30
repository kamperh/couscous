Couscous: Siamese Neural Networks for Representation Learning
=============================================================

Overview
--------
This is Theano neural network code, used to train Siamese networks for
representation learning. We have used the code here specifically to do
representation learning for whole-word speech segments using Siamese
convolutional neural networks (CNNs). The code here is the bare neural network
code, without feature extraction or evaluation code, or even an example script.
We hope to release a separate recipe illustrating the use of Siamese CNNs for
representation learning of word segments. For now, at least some of the test
functions in ``couscous/tests/`` should be helpful if you want to start to
unravel the current code. The name "Couscous" comes from the "cos cos^2" loss
used for training Siamese networks.


Dependencies
------------
- [Theano](http://deeplearning.net/software/theano/) and all its dependencies.
- [nose](https://nose.readthedocs.org/en/latest/)


Testing
-------
- Run ``make test`` to run unit tests.


Collaborators
-------------
- [Herman Kamper](http://www.kamperh.com/)
- [Weiran Wang](http://ttic.uchicago.edu/~wwang5/)
- [Karen Livescu](http://ttic.uchicago.edu/~klivescu/)

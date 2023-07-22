# Neural Network with Numpy

This is a fully functional two-layer neural network written in numpy.

The network consists of one hidden and one output layer followed by a softmax and cross entropy loss:
linear -> relu -> linear -> relu -> softmax -> cross entropy loss

To test the implementation the mnist dataset is used.

Example output:

Shape of X_train: (60000, 784)
Shape of y_train: (60000,)
Shape of X_test: (10000, 784)
Shape of y_test: (10000,)
(Epoch 1 / 10) avg train loss: 0.420522; avg validation loss: 0.413659
(Epoch 2 / 10) avg train loss: 0.295362; avg validation loss: 0.292576
(Epoch 3 / 10) avg train loss: 0.200191; avg validation loss: 0.201725
(Epoch 4 / 10) avg train loss: 0.177952; avg validation loss: 0.183086
(Epoch 5 / 10) avg train loss: 0.145278; avg validation loss: 0.157096
(Epoch 6 / 10) avg train loss: 0.121119; avg validation loss: 0.138721
(Epoch 7 / 10) avg train loss: 0.112205; avg validation loss: 0.133417
(Epoch 8 / 10) avg train loss: 0.100331; avg validation loss: 0.124652
(Epoch 9 / 10) avg train loss: 0.085152; avg validation loss: 0.114941
(Epoch 10 / 10) avg train loss: 0.073303; avg validation loss: 0.105111
final train acc: 0.977400; final validation acc: 0.967100
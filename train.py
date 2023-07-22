"""
Script tor training loop and gradient decent
"""
from model import *
from utils import *

# hyperparameters
batch_size = 128
epochs = 10
lr = 0.2
lr_decay = 0.5
decay_stepsize = 8

# get some data: download mnist train and test set
X_train, y_train = load_mnist('train')
X_test, y_test = load_mnist('t10k')

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)

# normalize input images
X_train = X_train.astype('float32') / 255 # scale [0,1]
X_test = X_test.astype('float32') / 255
X_train = X_train * 2 - 1 # scale [-1,1]
X_test = X_test * 2 - 1

# get model
model = TwoLayerNet(
    input_dim = np.prod(X_train.shape[1:]),
    hidden_dim = 100,
    out_dim = np.max(y_train) + 1
)

# training loop
for i in range(1, epochs + 1):
    for X_batch, y_batch in get_batches(X_train, y_train, batch_size):   

        # Compute loss and gradient
        loss, grads = model.loss(X_batch, y_batch)

        # Perform a parameter update
        for p, w in model.params.items():
            dw = grads[p]
            w -= lr * dw # vanilla gradient decent
            # model.params[p] = w
    
    # get full batch loss after epoch for train and test set
    train_loss, _ = model.loss(X_train, y_train, with_grads=False)
    validation_loss, _ = model.loss(X_test, y_test, with_grads=False) # test used as validation here for demonstration purposes

    print(
        "(Epoch %d / %d) avg train loss: %f; avg validation loss: %f"
        % (i, epochs, train_loss, validation_loss)
    )

    lr *= np.power(lr_decay, np.floor(i/decay_stepsize)) # lr decay

# Get final accuracies
preds_train, _ = model.forward(X_train)
preds_test, _ = model.forward(X_test)

train_correct = np.sum(np.argmax(preds_train, axis= 1) == y_train)
test_correct = np.sum(np.argmax(preds_test, axis= 1) == y_test)

train_acc = train_correct / X_train.shape[0]
test_acc = test_correct / X_test.shape[0]

print(
    "final train acc: %f; final validation acc: %f"
    % (train_acc, test_acc)
)
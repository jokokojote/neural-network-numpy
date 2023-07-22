"""
Collection of single layers for a simple neural network written in numpy
"""
import numpy as np

def linear_forward(x, w, b):
    """
   fully-connected layer: affine transformation

    N = batch size
    D = layer input size
    M = layer output size

    Inputs:
    - x: N batches of input data - shape (N, d_1, ..., d_k), is flattend to (N, D)
    - w: layer weights - shape (D, M)
    - b: layer biases -  shape (M,)

    Returns tuple:
    - out: layer output - shape (N, M)
    - cache: (x, w, b) for backpropagation
    """
    x_flattend = x.reshape(x.shape[0], np.prod(x.shape[1:])) # flatten input shape to (N, D, 1), that is 1-dimensional samples
    
    out = np.dot(x_flattend, w) + b # affine transformation w x + b

    cache = (x, w, b)
    return out, cache

def linear_backward(d_up, cache):
    """
    backward pass fc linear layer

    N = batch size
    M = layer output size

    Inputs:
    - d_up: Upstream derivatives - shape (N, M)
    - cache: Tuple:
      - x: forward pass input data - shape (N, d_1, ... d_k)
      - w: layer weights - (D, M)
      - b: layer biases - shape (M,)

    Returns tuple:
    - dx: Gradient wrt. x - shape (N, d1, ..., d_k)
    - dw: Gradient wrt. w - shape (D, M)
    - db: Gradient wrt. b - shape (M,)
    """
    x, w, b = cache
    x_flattend = x.reshape(x.shape[0], np.prod(x.shape[1:]))

    db = np.sum(d_up, axis=0) 
    dx = np.apply_along_axis(lambda losses: np.dot(w, losses), 1, d_up)
    dx = np.dot(d_up, w.T)
    dw = np.dot(x_flattend.T, d_up)

    dx = dx.reshape(x.shape) # reshape gradients to original input shape

    return dx, dw, db

def relu_forward(x):
    """
    forward pass for ReLU layer

    Input:
    - x: Inputs - any shape

    Returns tuple:
    - out: Output - same shape as x
    - cache: x for backpropagation
    """
    out = np.maximum(0, x)# = np.where(x>0, x, 0)

    cache = x
    return out, cache

def relu_backward(d_up, cache):
    """
    Backward pass for ReLU layer

    Input:
    - d_up: Upstream derivatives - any shape
    - cache: forward pass input x - same shape as dout

    Returns:
    - dx: Gradient wrt. x
    """
    dx = d_up * np.where(cache>0, 1, 0) # relu not differentiable at 0, can be set to 0 or 1 in practice

    return dx


def linear_relu_forward(x, w, b):
    """
    FC linear layer + ReLU combined

    Inputs:
    - x: Inputs
    - w, b: Parameters for affine layer

    Returns tuple:
    - out: Activations / ReLU out
    - cache: tuple: caches of single layers for backward pass
    """
    a, fc_cache = linear_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def linear_relu_backward(dout, cache):
    """
    Backward pass for the combined linear + relu layer

    Returns tuple:
    - dx: Gradient wrt. x - shape (N, d1, ..., d_k)
    - dw: Gradient wrt. w - shape (D, M)
    - db: Gradient wrt. b - shape (M,)
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = linear_backward(da, fc_cache)
    return dx, dw, db

def softmax(f):
    max_scores = np.max(f,axis=1)
    f -= max_scores.reshape(-1, 1) # shift f for numeric stability (max entry is 0)
    p = np.exp(f) / np.sum(np.exp(f), axis=1, keepdims=True)

    return p

def categorical_cross_entropy_loss(a, y):
    """
    Computes the loss and gradient for softmax classification.

    N = batch size
    C = nr. of target classes

    Inputs:
    - a: Input data - shape (N, C), activations of last layer / logits
    - y: Ground truth - shape (N,), y[i] is the labeled class for a[i]

    Returns a tuple of:
    - loss: Scalar giving the loss
    - da: Gradient of the loss with respect to a
    """
    N = a.shape[0]

    # get predicted probabilities - TODO: log softmax would be better for numerical stability
    p = softmax(a)

    epsilon = 1e-8 # avoid log(0) -> add small constant
    loss = -np.log(p[np.arange(N), y] + epsilon) # categorical CE loss for one-hot targets / negative log likelihood
    loss = np.sum(loss) / N

    # get full sparse one hot vectors for gradient caluclation
    y_onehot = np.zeros_like(a)
    y_onehot[range(N), y] = 1.

    # gradient
    da = p - y_onehot # overall derivate of cross_entropy(softmax(a)), dL/da = dL/dp * dp/da
    da = da / N # handle batching
    
    return loss, da
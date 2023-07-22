from layers import *
"""
Definition of a complete two layer neural network with numpy
"""

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    cross_entropy_loss

    Architecure: linear - relu - linear - relu

    Learnable parameters of the model are stored in dictionary self.params
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        weight_scale=1e-2,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: size of the input
        - hidden_dim: size of hidden layer
        - num_classes: number of target classes
        - weight_scale: standard deviation for gaußian weights initilization
        """
        weights_l1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        biases_l1 = np.zeros(hidden_dim)
        weights_l2 = np.random.normal(0, weight_scale, (hidden_dim, out_dim))
        biases_l2 = np.zeros(out_dim)

        self.params = {'W1': weights_l1,
                       'b1': biases_l1,
                       'W2': weights_l2,
                       'b2': biases_l2}

    def forward(self, X):
        """
        Feed-forward pass
        
        Inputs:
        - X: Input data - shape (N, D), N = batch size, D = input dimension.
        
        Returns:
        - scores: Output scores - shape (N, C), C = number of target classes
        - cache: Tuple, caching objects ((cache1, cache2)) for backpropagation. cache1 and cache2 are caches 
                from linear_relu_forward calls for hidden layer and output layer respectively.
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

         #forward passes
        hidden, cache1 = linear_relu_forward(X, W1, b1)
        scores, cache2 = linear_relu_forward(hidden, W2, b2)

        return scores, (cache1, cache2)
    
    def backward(self, d_out, cache):
        """
        Backward pass
        
        Inputs:
        - d_out: Upstream derivatives (from loss function) - shape (N, C), N = batch size, C = number of classes
        - cache: Tuple containing caching objects ((cache1, cache2)) from forward pass. cache1 and cache2 are caches 
                from linear_relu_forward calls for hidden layer and output layer respectively
        
        Returns:
        - grads: Dictionary mapping parameter names to gradients of those parameters - keys ['W1', 'b1', 'W2', 'b2']
        """
        cache1, cache2 = cache

        dx2, dw2, db2 = linear_relu_backward(d_out, cache2)
        dx1, dw1, db1 = linear_relu_backward(dx2, cache1)
        
        grads = {'W1': dw1,
                 'W2': dw2,
                 'b1': db1,
                 'b2': db2}
        
        return grads

    def loss(self, X, y, with_grads = True):
        """
        Computes loss and gradient for a minibatch of data, that is includes forward and backward pass.

        Inputs:
        - X: input data - shape (N, d_1, ..., d_k)
        - y: labels - shape (N,). y[i] gives the label for X[i].
        - with_grads: calculate gradient / do forward pass only

        Returns: tuple:
        - loss: Scalar value giving the loss
        - grads: gradients of the loss with respect to the model parameters OR none if with_grads = FALSE
        """
        #forward pass
        scores, cache = self.forward(X)

        # loss
        loss, d_out = categorical_cross_entropy_loss(scores, y)
        
        if not with_grads: # just forwards pass, no gradients needed
            return loss, None

        # backward pass
        grads = self.backward(d_out, cache)

        return loss, grads
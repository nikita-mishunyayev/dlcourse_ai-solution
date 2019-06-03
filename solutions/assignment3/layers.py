import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad

def softmax(predictions):
        predictions -= np.max(predictions)
        probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    
        return probs


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    preds = predictions.copy()
    probs = softmax(preds)
    
    loss = - np.log(probs[np.arange(len(probs)), target_index])
    loss = loss.mean()
    mask = np.zeros_like(predictions)

    # mask and dprediction for (N) shape predictions
    if predictions.shape == (len(predictions), ):
        mask[target_index] = 1
        dprediction = - (mask - probs)
        
    # mask and dprediction for (batch_size, N) shape predictions
    else:
        mask[np.arange(len(mask)), target_index] = 1
        dprediction = - (mask - probs) / (len(mask))

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        result = X.copy()
        result[result < 0] = 0
        
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = (self.X > 0) * d_out
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        W = self.W.value
        B = self.B.value
        result = np.dot(X, W) + B
        
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        X = self.X
        W = self.W.value
        
        d_W = np.dot(X.T, d_out)
        d_B = np.dot(np.ones((X.shape[0], 1)).T, d_out)
        d_X = np.dot(d_out, W.T)
        
        self.W.grad += d_W
        self.B.grad += d_B

        return d_X

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

    def forward(self, X):
        padding = self.padding
        if padding > 0:
            X = np.insert(X, 0, np.zeros([ padding]), axis=2)
            X = np.insert(X, X.shape[2], np.zeros([ padding]), axis=2)
            X = np.insert(X, 0, np.zeros([ padding]), axis=1)
            X = np.insert(X, X.shape[1], np.zeros([ padding]), axis=1)
        
        self.X = X
        filter_size = self.filter_size
        out_channels = self.out_channels
        batch_size, height, width, in_channels = X.shape
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below

        out_height = height - (filter_size - 1)
        out_width = width - (filter_size - 1)
        result = np.zeros((batch_size, out_height, out_width, out_channels))
        
        W_reshaped = self.W.value.reshape((-1, out_channels))
        B = self.B.value
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_after_kernel = X[:, y:y+filter_size, x:x+filter_size, :]
                X_after_kernel = X_after_kernel.reshape((batch_size, -1))
                out = np.dot(X_after_kernel, W_reshaped) + B
                #print(np.dot(X_after_kernel, W_reshaped))
                #print(B)
                #print(out, out.shape)
                result[:, y, x, :] = out
                
        return result
    
    def backward(self, d_out):        
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        X = self.X
        d_X = np.zeros_like(X)
        
        padding = self.padding
        filter_size = self.filter_size
        batch_size, height, width, in_channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
        
        W_reshaped = self.W.value.reshape((-1, out_channels))
        ###d_B = np.sum(d_out, (0, 1, 2))
        ###self.B.grad += d_B

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                point = d_out[:, y, x, :]
                
                X_after_kernel = X[:, y:y+filter_size, x:x+filter_size, :]
                X_after_kernel_reshaped = X_after_kernel.reshape((batch_size, -1))
                
                d_W = np.dot(X_after_kernel_reshaped.T, point)
                d_W = d_W.reshape((filter_size, filter_size, in_channels, out_channels))
                
                d_B = np.dot(np.ones((batch_size, )).T, point)
                
                d_X_before_kernel = np.dot(point, W_reshaped.T)
                d_X_before_kernel = d_X_before_kernel.reshape((batch_size, filter_size, filter_size, in_channels))
                
                self.W.grad += d_W
                self.B.grad += d_B
                d_X[:, y:y+filter_size, x:x+filter_size, :] += d_X_before_kernel
                
        d_X = d_X[:, padding:height-padding, padding:width-padding, :]
        return d_X

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on output x/y dimension
        pool_size = self.pool_size
        stride = self.stride
        self.X = X
        batch_size, height, width, in_channels = X.shape

        
        out_height = int(np.floor((height - pool_size) / stride)) + 1
        out_width = int(np.floor((width - pool_size) / stride)) + 1
        result = np.zeros((batch_size, out_height, out_width, in_channels))
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(in_channels):
                        y_source = y * stride
                        x_source = x * stride
                        pool = X[batch, y_source:y_source+pool_size, x_source:x_source+pool_size, channel]
                        maximum = np.max(pool)
                        result[batch, y, x, channel] = maximum
        
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        pool_size = self.pool_size
        stride = self.stride
        X = self.X
        batch_size, height, width, in_channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_X = np.zeros_like(X)
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(in_channels):
                        mask = np.zeros((pool_size, pool_size))
                        y_source = y * stride
                        x_source = x * stride
                        pool = X[batch,
                                 y_source:np.minimum(y_source+pool_size, height),
                                 x_source:np.minimum(x_source+pool_size, width), channel]
                        
                        maximum = np.max(pool)
                        max_count = np.count_nonzero(pool == maximum)
                        argmax = np.argwhere(pool==maximum)
                        mask[argmax[:,0], argmax[:,1]] = d_out[batch, y, x, channel] / max_count
                        
                        d_X[batch, y_source:y_source+pool_size, x_source:x_source+pool_size, channel] += mask
                        
        return d_X

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, -1))

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X.shape
        
        return d_out.reshape((batch_size, height, width, channels))

    def params(self):
        # No params!
        return {}

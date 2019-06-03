import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        width, height, n_input_channels = input_shape
        kernel_size = 3
        padding = 1
        conv_stride = 1
        pooling_stride = 4
        filter_size = 4
        
        conv1_output = (width - kernel_size + 2*padding) / conv_stride + 1
        pooling1_output = (conv1_output - filter_size) / pooling_stride + 1
        conv2_output = (pooling1_output - kernel_size + 2*padding) / conv_stride + 1
        pooling2_output = (conv2_output - filter_size) / pooling_stride + 1
        fc_input = int(pooling2_output * pooling2_output * conv2_channels)
        
        self.Sequential = [
            ConvolutionalLayer(n_input_channels, conv1_channels, kernel_size, padding),
            ReLULayer(),
            MaxPoolingLayer(filter_size, pooling_stride),
            ConvolutionalLayer(conv1_channels, conv2_channels, kernel_size, padding),
            ReLULayer(),
            MaxPoolingLayer(filter_size, pooling_stride),
            Flattener(),
            FullyConnectedLayer(fc_input, n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params = self.params()
        for p in params:
            params[p].grad = np.zeros_like(params[p].value)

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        X_copy = X.copy()
        for layer in self.Sequential:
            X_copy = layer.forward(X_copy)
        
        predictions = X_copy
        loss, d_predictions = softmax_with_cross_entropy(predictions, y)
        
        for layer in reversed(self.Sequential):
            d_predictions = layer.backward(d_predictions)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        for layer in self.Sequential:
            X = layer.forward(X)
        
        probs = softmax(X)
        predictions = np.argmax(probs, axis=1)
        
        return predictions

    def params(self):
        # TODO Implement aggregating all of the params
        result = {}
        for layer_number in range(len(self.Sequential)):
            for i in self.Sequential[layer_number].params():
                result[str(layer_number) + '_' + i] = self.Sequential[layer_number].params()[i]

        return result

import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization
from assignments_in_progress.assignment1.linear_classifer import softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.Linear1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.Linear2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        params = self.params()
        
        W1 = params['W1']
        B1 = params['B1']
        W2 = params['W2']
        B2 = params['B2']
        
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        forward_linear1 = self.Linear1.forward(X)
        forward_relu = self.ReLU.forward(forward_linear1)
        forward_linear2 = self.Linear2.forward(forward_relu)
        
        predictions = forward_linear2
        loss, d_predictions = softmax_with_cross_entropy(predictions, y)
        backward_linear2 = self.Linear2.backward(d_predictions)
        backward_relu = self.ReLU.backward(backward_linear2)
        backward_linear1 = self.Linear1.backward(backward_relu)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)
        
        l2_loss = l2_W1_loss + l2_W2_loss + l2_B1_loss + l2_B2_loss
        loss += l2_loss
        
        W1.grad += l2_W1_grad
        B1.grad += l2_B1_grad
        W2.grad += l2_W2_grad
        B2.grad += l2_B2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        output = self.Linear1.forward(X)
        output = self.ReLU.forward(output)
        output = self.Linear2.forward(output)
        
        probs = softmax(output)
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
        result = {
            'W1': self.Linear1.params()['W'],
            'B1': self.Linear1.params()['B'],
            'W2': self.Linear2.params()['W'],
            'B2': self.Linear2.params()['B']
        }

        return result

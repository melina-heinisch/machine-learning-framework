from enum import Enum
from machine_learning_framework import mse, mse_derivative, cross_entropy, categorical_cross_entropy, softmax
import numpy as np

class NeuralNet:

    class Type(Enum):
        REGRESSION = 1
        BINARY_CLASSIFICATION = 2
        MUTILCLASS_CLASSIFICATION = 3

    def __init__(self, layer_sizes, type, activation_fn, optimizer_fn, weights = None, biases = None):
        # Set variables class wide
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        self.activation_fn = activation_fn
        self.optimizer_fn = optimizer_fn
        self.type = type
        self.use_softmax = False

        if type == self.Type.REGRESSION:
            self.loss_fn = mse
        elif type == self.Type.BINARY_CLASSIFICATION:
            self.loss_fn = cross_entropy
        elif type == self.Type.MUTILCLASS_CLASSIFICATION:
            self.loss_fn = categorical_cross_entropy
            self.use_softmax = True

        # Initialize weights and biases
        if weights is None or biases is None:
            self.init_weights_biases()
        else:
            self.weights = weights
            self.biases = biases


    def init_weights_biases(self):
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01 for i in range(self.n_layers)]
        self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.n_layers)]
    
    def forward_propagation(self, X):
        a = [X]  # Collects activations for each layer
        z = []   # Collects weighted inputs for each layer

        for i in range(self.n_layers):
            z.append(a[i].dot(self.weights[i]) + self.biases[i])
            if (i ==self.n_layers - 1) and self.use_softmax:
                a.append(softmax(z[-1]))
            else:
                a.append(self.activation_fn(z[-1]))

        return a, z

    def backward_propagation(self, Y, a, z):
        O = a[-1]
        if self.type == self.Type.REGRESSION:
            first_delta = mse_derivative(O, Y)
        elif self.type == self.Type.BINARY_CLASSIFICATION:
            first_delta = (-(Y/O) + ((1-Y)/(1-O))) * self.activation_fn(z[-1], True)
        elif self.type == self.Type.MUTILCLASS_CLASSIFICATION:
            first_delta = O - Y
        
        delta = [first_delta]  # List of errors for each layer, starting with output layer
        dW = []  # Collects weight gradients for each layer
        db = []  # Collects bias gradients for each layer

        # reminder: range(start, stop, step)
        for i in range(self.n_layers-1, -1, -1):
            dW.append(a[i].T.dot(delta[-1]))
            db.append(np.sum(delta[-1], axis=0, keepdims=True))
            if i > 0:
                delta.append(delta[-1].dot(self.weights[i].T) * self.activation_fn(z[i-1], True))

        # Since we went backwards we need to reverse here to get right order
        dW.reverse()
        db.reverse()

        return dW, db

    def train(self,X_train, y_train, alpha, iterations, lambda_, X_validate = None, y_validate = None):
        # First we need to initialize several variables
        m = y_train.shape[0]
        error_history_train = []
        error_history_validate = []

        for iteration in range(iterations):
            # Perform forward propagation and compute loss
            a, z = self.forward_propagation(X_train)
            loss = self.loss_fn(y_train, a[-1])
            regularization = (lambda_/(2*m)) * np.sum(np.sum(w**2) for w in self.weights)
            reg_loss = loss + regularization

            # Get validation loss
            if(X_validate is not None and y_validate is not None):
                a_validate = self.predict(X_validate)
                loss_validate = self.loss_fn(y_validate,a_validate)
                regularization_validate = (lambda_/(2*y_validate.shape[0])) * np.sum(np.sum(w**2) for w in self.weights)
                reg_loss_validate = loss_validate + regularization_validate

            # Perform backward propagation and compute gradients
            dW, dB = self.backward_propagation(y_train, a, z)

            # Update the weights and biases
            self.weights, self.biases = self.optimizer_fn(self.weights, self.biases, dW, dB, alpha, lambda_, m)

            # Save the loss every 10 iterations
            if iteration % 10 == 0:
                error_history_train.append(reg_loss)
                if 'reg_loss_validate' in locals():
                    error_history_validate.append(reg_loss_validate)

        return self.weights, self.biases, error_history_train, error_history_validate

    # The predict method is very similar to the forward propagation, the
    # difference being that we do not need to save z and we return only the output
    def predict(self, X):
        a = [X]
        for i in range(self.n_layers):
            z = a[i].dot(self.weights[i]) + self.biases[i]
            if (i == self.n_layers - 1) and self.use_softmax:
                a.append(softmax(z))
            else:
                a.append(self.activation_fn(z))

       # if self.type == self.Type.MUTILCLASS_CLASSIFICATION:
            # "argmax" returns the indices of the maximum values in a row, 
            # so the index of the class that is being predicted by the model
            # We only need this for multiclass cassification since it uses one-hot encoding
       #     return np.argmax(a[-1], axis=1)
       # else:
        return a[-1]


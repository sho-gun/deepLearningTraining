import sys, os
sys.path.append(os.pardir)
import numpy as np
import cupy as cp
from module.ActivationFunction import sigmoid, softmax
from module.LossFunction import cross_entropy_error
from module.Gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, enable_gpu=False):
        self.enable_gpu = enable_gpu

        # init weights
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        if self.enable_gpu:
            print('GPU enabled.')
            self.params['W1'] = cp.asarray(self.params['W1'])
            self.params['b1'] = cp.asarray(self.params['b1'])
            self.params['W2'] = cp.asarray(self.params['W2'])
            self.params['b2'] = cp.asarray(self.params['b2'])

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        if self.enable_gpu:
            a1 = cp.dot(x, W1) + b1
            z1 = sigmoid(a1)
            a2 = cp.dot(z1, W2) + b2
            y = softmax(a2)
        else:
            a1 = np.dot(x, W1) + b1
            z1 = sigmoid(a1)
            a2 = np.dot(z1, W2) + b2
            y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)

        if self.enable_gpu:
            y = cp.argmax(y, axis=1)
            t = cp.argmax(t, axis=1)

            accuracy = cp.sum(y == t) / float(x.shape[0])

        else:
            y = np.argmax(y, axis=1)
            t = np.argmax(t, axis=1)

            accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

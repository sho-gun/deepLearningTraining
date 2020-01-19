import sys, os
sys.path.append(os.pardir)
import numpy as np
from module.ActivationFunction import softmax
from module.LossFunction import cross_entropy_error

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

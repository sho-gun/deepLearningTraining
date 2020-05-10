import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)

    y = step_function(x)
    # y = sigmoid(x)
    # y = relu(x)
    # y = identity_function(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

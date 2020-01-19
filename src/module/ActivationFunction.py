import numpy as np
import cupy as cp
import matplotlib.pylab as plt

def step_function(x):
    y = x > 0
    if type(x).__module__ == np.__name__:
        return y.astype(np.int)
    else:
        return y.astype(cp.int)

def sigmoid(x):
    if type(x).__module__ == np.__name__:
        return 1 / (1 + np.exp(-x))
    else:
        return 1 / (1 + cp.exp(-x))

def relu(x):
    if type(x).__module__ == np.__name__:
        return np.maximum(0, x)
    else:
        return cp.maximum(0, x)

def identity_function(x):
    return x

def softmax(x):
    if type(x).__module__ == np.__name__:
        c = np.max(x)
        exp_x = np.exp(x - c)
        return exp_x / np.sum(exp_x)
    else:
        c = cp.max(x)
        exp_x = cp.exp(x - c)
        return exp_x / cp.sum(exp_x)

if __name__ == '__main__':
    x = cp.arange(-5.0, 5.0, 0.1)

    y = step_function(x)
    # y = sigmoid(x)
    # y = relu(x)
    # y = identity_function(x)

    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

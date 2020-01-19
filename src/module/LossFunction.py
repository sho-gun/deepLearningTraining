import numpy as np
import cupy as cp

def mean_squared_error(y, t):
    if type(y).__module__ == np.__name__:
        return 0.5 * np.sum((y-t) ** 2)
    else:
        return 0.5 * cp.sum((y-t) ** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7

    if type(y).__module__ == np.__name__:
        return -np.sum(t * np.log(y + delta)) / batch_size
    else:
        return -cp.sum(t * cp.log(y + delta)) / batch_size

import numpy as np

def numerical_diff(func, x):
    h = 1e-4
    return (func(x+h) - func(x-h)) / (2*h)

def numerical_gradient(func, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = func(x)

        x[idx] = tmp_val - h
        fxh2 = func(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        it.iternext()

    return grad

import os
import pickle
import numpy as np
from module.ActivationFunction import sigmoid, softmax
from util.mnist import load_mnist

def getData():
    (xTrain, tTrain), (xTest, tTest) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return xTest, tTest

def initNetwork():
    with open(os.path.join('weight', 'sample_weight.pkl'), 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

if __name__ == '__main__':
    x, t = getData()
    network = initNetwork()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print('Accuracy:' + str(float(accuracy_cnt) / len(x)))

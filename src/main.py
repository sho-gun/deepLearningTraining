import sys
import numpy as np
import cupy as cp
from util.mnist import load_mnist
from network.TwoLayerNet import TwoLayerNet
import matplotlib.pyplot as plt

USE_GPU = False

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1 and args[1] == '-g':
        USE_GPU = True

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

    if USE_GPU:
        # convert into cupy array
        x_train = cp.asarray(x_train)
        t_train = cp.asarray(t_train)
        x_test = cp.asarray(x_test)
        t_test = cp.asarray(t_test)

    train_loss_list = []

    # hparams
    iters_num = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, enable_gpu=USE_GPU)

    for i in range(iters_num):
        print('Iter', i)

        # get training batch
        if USE_GPU:
            batch_mask = cp.random.choice(train_size, batch_size)
        else:
            batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # get gradient
        grad = network.numerical_gradient(x_batch, t_batch)

        # update params
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # save
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        print('Loss:', loss)

    # show graph
    x = np.arange(0, iters_num, 1)
    y = np.array(train_loss_list)
    plt.plot(x, y)
    plt.title('Train Loss')
    plt.show()

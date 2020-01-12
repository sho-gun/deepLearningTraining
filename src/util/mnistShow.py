import sys
import os
import numpy as np
from mnist import load_mnist
from PIL import Image

def imgShow(img):
    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()

if __name__ == '__main__':
    (xTrain, tTrain), (xTest, tTest) = load_mnist(flatten=True, normalize=False)

    img = xTrain[0]
    label = tTrain[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    imgShow(img)

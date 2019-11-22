import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _load(filepath):
    x = []
    y = []
    with open(filepath, 'r') as fp:
        for line in fp:
            vec = [int(i) for i in line.split(',')]
            x.append(vec[1:])
            y.append(vec[0])

        return np.array(x), np.array(y)


def _scale(x):
    return MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)


def load_train():
    x, y = _load("data/mnist_train.txt")
    return _scale(x), y


def load_test():
    x, y = _load("data/mnist_test.txt")
    return _scale(x), y

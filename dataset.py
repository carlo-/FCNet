#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import utilities as util
import numpy as np

DATASET_FOLDER = '../dataset/'

def load_cifar10(batch='data_batch_1', limit_N=None, limit_d=None, zero_mean=True):

    # Unpickled raw dataset.py
    dict = util.unpickle(DATASET_FOLDER + 'cifar-10/' + batch)

    # Vectorized images (3072*10000)
    X = dict[b'data'].T / 255.0

    # Labels
    y = np.array(dict[b'labels'])

    if limit_N is not None:
        X = X[:, :limit_N]
        y = y[:limit_N]

    if limit_d is not None:
        X = X[:limit_d, :]

    # Number of images
    N = X.shape[1]

    # Number of classes
    K = 10

    # One-hot representation of the labels (K*N)
    Y = np.zeros((y.size, K))
    Y[np.arange(y.size), y] = 1
    Y = Y.T

    if zero_mean:
        X = subtract_mean(X)

    # Dimension of each image
    # d = 32*32*3

    return (X, Y, y)

def load_multibatch_cifar10(batches=None, limit_d=None, zero_mean=True):
    X, Y, y = ([], [], [])
    if batches is None:
        batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for b in batches:
        Xi, Yi, yi = load_cifar10(b, limit_d=limit_d, zero_mean=zero_mean)
        X.append(Xi)
        Y.append(Yi)
        y.append(yi)
    X = np.concatenate(X, axis=1)
    Y = np.concatenate(Y, axis=1)
    y = np.concatenate(y)
    return X, Y, y

def subtract_mean(X):
    return X - np.mean(X, axis=1).reshape(-1,1)

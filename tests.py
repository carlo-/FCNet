#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import numpy as np
from model import Net
import utilities as util
import dataset


def test_two_layers():

    np.random.seed(42)
    X, Y, y = dataset.load_multibatch_cifar10()
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, K]
    gd_params = {
        'eta': 0.022661,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.6,
        'decay_rate': 0.93,
        'lambda': 0.000047,
        'plateau_guard': -0.0009,
        'batch_normalize': False
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    # costs, test_costs = (r['costs'], r['test_costs'])
    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))


def test_three_layers():

    np.random.seed(42)
    X, Y, y = dataset.load_multibatch_cifar10()
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, 30, K]

    gd_params = {
        'eta': 0.01,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.0,
        'decay_rate': 1.0,
        'lambda': 0.0,
        'batch_normalize': True
    }

    gd_params = {
        'eta': 0.015661,
        'batch_size': 100,
        'epochs': 20,
        'gamma': 0.6,
        'decay_rate': 0.93,
        'lambda': 0.000047,
        'plateau_guard': -0.0009,
        'overfitting_guard': 0.0,
        'batch_normalize': True
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    # costs, test_costs = (r['costs'], r['test_costs'])
    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

def test_import():

    filepath = './model_epoch_20.pkl'
    net = Net.import_model(filepath)

    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    acc = net.compute_accuracy(X_test, y_test)
    print('test acc', acc)

if __name__ == "__main__":
    # test_two_layers()
    test_three_layers()
    # test_import()

#
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#


## Imports ##

import numpy as np
from timeit import default_timer as timer
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
#import pandas as pd


## Utilities ##

def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def play_bell():
    os.system('afplay /System/Library/Sounds/Ping.aiff')


## Reading the dataset ##

def load_cifar10(batch='data_batch_1', limit_N=None, limit_d=None, zero_mean=True):
    # Unpickled raw dataset
    dict = unpickle('./dataset/cifar-10/' + batch)

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

def load_multibatch_cifar10(batches=['data_batch_1'], limit_d=None, zero_mean=True):
    X, Y, y = ([], [], [])
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


## Model ##

class Net:

    def __init__(self, network_sizes, descent_params):
        self.network_sizes = network_sizes
        self.n_layers = len(network_sizes)-1
        self.descent_params = descent_params
        self.lamb = descent_params.get('lambda', 0.0)
        self.Ws, self.bs = self.initial_theta()

    def initial_theta(self):
        mu = 0.0
        sigma = 0.001
        Ws, bs = [], []
        for i in range(self.n_layers):
            lhs, rhs = self.network_sizes[i], self.network_sizes[i+1]
            Wi = sigma * np.random.randn(rhs,lhs) + mu
            bi = np.zeros([rhs,1])
            Ws.append(Wi)
            bs.append(bi)
        return Ws, bs

    def softmax(self, s, axis=0):
        exp_s = np.exp(s)
        exp_sum = np.sum(exp_s, axis=axis)
        return exp_s / exp_sum

    def forward(self, X):
        # Each column of P contains the probability for each
        # label for the image in the corresponding column of X
        # (P has size K*N)

        # Hs are the outputs of each layer, including the input layer, excluding the output
        # Hs[0] = X

        Hi = X #.copy()
        si = None
        Hs = []

        for i in range(self.n_layers):
            Hs.append(Hi)
            Wi, bi = self.Ws[i], self.bs[i]
            si = Wi @ Hi + bi
            Hi = np.maximum(0.0, si)

        P = self.softmax(si)
        return Hs, P

    def cross_entropy_loss(self, X, Y):
        N = X.shape[1]
        _, P = self.forward(X)
        loss = -Y * np.log(P)
        return np.sum(loss) / N

    def compute_cost(self, X, Y):
        # Regularization term
        L_2 = np.sum([np.sum(Wi ** 2) for Wi in self.Ws])

        # Cross-entropy loss
        ce_loss = self.cross_entropy_loss(X, Y)

        return ce_loss + self.lamb * L_2

    def classify(self, X):
        _, P = self.forward(X)
        return np.argmax(P, axis=0)

    def compute_accuracy(self, X, y):
        y_star = self.classify(X)
        # Count the number of correctly classified samples
        correct = np.sum([y_star == y])
        N = X.shape[1]
        return float(correct) / N

    def compute_gradients_fast(self, X, Y, P, Hs):

        N = X.shape[1]
        G = (P - Y)
        grads_W, grads_b = [], []

        for j in range(self.n_layers):

            i = self.n_layers -1 -j # Reversed
            Hi, Wi = Hs[i], self.Ws[i]

            grad_bi = np.mean(G, axis=1).reshape(-1, 1)
            grad_Wi = (G @ Hi.T) / N + (2 * self.lamb * Wi)

            grads_b.append(grad_bi)
            grads_W.append(grad_Wi)

            G = (G.T @ Wi).T
            G[Hi <= 0] = 0.0

        grads_W.reverse()
        grads_b.reverse()

        return grads_W, grads_b

    def train(self, X, Y, X_test, Y_test, silent=False):

        tick_t = timer()

        params = self.descent_params
        batch_size = params.get('batch_size', 100)
        epochs = params.get('epochs', 40)
        eta = params.get('eta', 0.01)
        gamma = params.get('gamma', 0.0)
        decay_rate = params.get('decay_rate', 1.0)
        lamb = params.get('lambda', 0.0)
        plateau_guard = params.get('plateau_guard', None)

        N = X.shape[1]
        batches = N // batch_size
        Ws, bs = self.Ws, self.bs

        # Prepare the momentum vectors
        v_W = [np.zeros(a.shape) for a in Ws]
        v_b = [np.zeros(a.shape) for a in bs]

        # Convert Y (one-hot) into a normal label representation
        y = np.argmax(Y, axis=0)

        # Keep track of the performance at each epoch
        costs = [self.compute_cost(X, Y)]
        losses = [self.cross_entropy_loss(X, Y)]
        accuracies = [self.compute_accuracy(X, y)]
        test_costs = None
        test_losses = None
        test_accuracies = None
        times = []
        speed = []
        test_speed = []

        y_test = np.argmax(Y_test, axis=0)
        test_costs = [self.compute_cost(X_test, Y_test)]
        test_losses = [self.cross_entropy_loss(X_test, Y_test)]
        test_accuracies = [self.compute_accuracy(X_test, y_test)]

        # For each epoch
        for e in range(1, epochs + 1):

            tick_e = timer()

            # For each mini batch
            for i in range(batches):

                # Extract batch
                i_beg = i * batch_size
                i_end = (i + 1) * batch_size
                X_batch = X[:, i_beg:i_end]
                Y_batch = Y[:, i_beg:i_end]

                # Compute gradients
                Hs, P = self.forward(X_batch)
                grads_W, grads_b = self.compute_gradients_fast(X_batch, Y_batch, P, Hs)

                # Update W and b
                for j in range(len(Ws)):
                    v_W[j] = gamma * v_W[j] + eta * grads_W[j]
                    v_b[j] = gamma * v_b[j] + eta * grads_b[j]
                    Ws[j] -= v_W[j]
                    bs[j] -= v_b[j]

            # Apply the decay rate to eta
            eta *= decay_rate

            # Keep track of the performance at each epoch
            costs.append(self.compute_cost(X, Y))
            losses.append(self.cross_entropy_loss(X, Y))
            accuracies.append(self.compute_accuracy(X, y))

            test_costs.append(self.compute_cost(X_test, Y_test))
            test_losses.append(self.cross_entropy_loss(X_test, Y_test))
            test_accuracies.append(self.compute_accuracy(X_test, y_test))

            dJ = costs[-1] - costs[-2]
            dJ_star = test_costs[-1] - test_costs[-2]

            speed.append(dJ)
            test_speed.append(dJ_star)
            mean_dJ_star = np.mean(test_speed[-2:])

            if eta > 0.001 and plateau_guard is not None and mean_dJ_star >= plateau_guard:
                if not silent:
                    print('Plateau reached, adjusting eta...')
                eta /= 10.0

            if not silent:
                tock_e = timer()
                interval = tock_e - tick_e
                times.append(interval)
                rem = (epochs - e) * np.mean(times[-3:])
                print('===> Epoch[{}]: {}s remaining, {} dJ, {} dJ*'.format(e, int(round(rem)), round(dJ, 5), round(dJ_star, 5)))

        if not silent:
            tock_t = timer()
            print("Done. Took ~{}s".format(round(tock_t - tick_t)))

        return {
            'costs': costs,
            'test_costs': test_costs,
            'losses': losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'test_accuracies': test_accuracies,
            'speed': speed,
            'test_speed': test_speed
        }

def debug_two_layers():
    np.random.seed(42)
    X, Y, y = load_multibatch_cifar10(batches=[
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'
    ])
    X_test, Y_test, y_test = load_cifar10(batch='test_batch', limit_N=None)
    K, d = (Y.shape[0], X.shape[0])
    net_sizes = [d, 50, K]
    gd_params = {
        'eta': 0.022661,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.6,
        'decay_rate': 0.93,
        'lambda': 0.000047,
        'plateau_guard': -0.0009
    }
    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    costs, test_costs = (r['costs'], r['test_costs'])
    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

# debug_two_layers()

#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import os
import numpy as np


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


def play_bell():
    os.system('afplay /System/Library/Sounds/Ping.aiff')


def compute_gradients_num_slow(X, Y, W, b, lamb, h, cost_fn):
    assert len(W) == len(b)
    grads_W, grads_b = ([], [])

    for j in range(len(W)):

        W_try = [a.copy() for a in W]
        b_try = [a.copy() for a in b]
        grad_W = np.zeros(W_try[j].shape)
        grad_b = np.zeros([W_try[j].shape[0], 1])

        for i in range(b_try[j].size):
            b_try[j][i] = b_try[j][i] - h
            c1 = cost_fn(X, Y, W, b_try, lamb)
            b_try[j][i] = b_try[j][i] + h

            b_try[j][i] = b_try[j][i] + h
            c2 = cost_fn(X, Y, W, b_try, lamb)
            b_try[j][i] = b_try[j][i] - h

            grad_b[i] = (c2 - c1) / (2 * h)

        for i in range(W_try[j].size):
            W_try[j].itemset(i, W_try[j].item(i) - h)
            c1 = cost_fn(X, Y, W_try, b, lamb)
            W_try[j].itemset(i, W_try[j].item(i) + h)

            W_try[j].itemset(i, W_try[j].item(i) + h)
            c2 = cost_fn(X, Y, W_try, b, lamb)
            W_try[j].itemset(i, W_try[j].item(i) - h)

            grad_W.itemset(i, (c2 - c1) / (2 * h))

        grads_W.append(grad_W)
        grads_b.append(grad_b)

    return grads_W, grads_b


def compute_gradients_num(X, Y, W, b, lamb, h, cost_fn):
    assert len(W) == len(b)
    c = cost_fn(X, Y, W, b, lamb)
    grads_W, grads_b = ([], [])

    for j in range(len(W)):

        W_try = [a.copy() for a in W]
        b_try = [a.copy() for a in b]
        grad_W = np.zeros(W_try[j].shape)
        grad_b = np.zeros([W_try[j].shape[0], 1])

        for i in range(b_try[j].size):
            b_try[j][i] = b_try[j][i] + h
            c2 = cost_fn(X, Y, W, b_try, lamb)
            b_try[j][i] = b_try[j][i] - h

            grad_b[i] = (c2 - c) / h

        for i in range(W_try[j].size):
            W_try[j].itemset(i, W_try[j].item(i) + h)
            c2 = cost_fn(X, Y, W_try, b, lamb)
            W_try[j].itemset(i, W_try[j].item(i) - h)

            grad_W.itemset(i, (c2 - c) / h)

        grads_W.append(grad_W)
        grads_b.append(grad_b)

    return grads_W, grads_b


def relative_err(a,b,eps=1e-12):
    assert a.shape == b.shape
    return np.abs(a-b) / np.maximum(eps, np.abs(a)+np.abs(b))


def compare_dthetas(lhs,rhs):
    assert len(lhs) == len(rhs)
    grads_W_lhs, grads_b_lhs = lhs
    grads_W_rhs, grads_b_rhs = rhs
    assert len(grads_W_lhs) == len(grads_W_rhs) and len(grads_b_lhs) == len(grads_b_rhs)
    return np.array(
        [np.mean(relative_err(grads_W_lhs[i],grads_W_rhs[i])) for i in range(len(grads_W_lhs))] +
        [np.mean(relative_err(grads_b_lhs[i],grads_b_rhs[i])) for i in range(len(grads_b_lhs))]
    ).reshape(-1,1)

#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from os.path import exists
from utilities import unpickle
from model import Net
import dataset


FINE_SEARCH_CONFIGS_PATH = '../configs_fine.pkl'
FINE_SEARCH_RESULTS_PATH = '../results_fine.pkl'
COARSE_SEARCH_CONFIGS_PATH = '../configs_coarse.pkl'
COARSE_SEARCH_RESULTS_PATH = '../results_coarse.pkl'


def _search_worker(net_sizes, config, X, Y, X_test, Y_test):
    net = Net(net_sizes, config)
    return net.train(X.copy(), Y.copy(), X_test.copy(), Y_test.copy(), silent=True)


def _get_configs(default_params, n, min_eta, max_eta, min_lambda, max_lambda):
    configs = []
    for i in range(n):
        p = default_params.copy()

        min_e, max_e = (np.log10(min_eta), np.log10(max_eta))
        e = min_e + (max_e - min_e) * np.random.rand()
        p['eta'] = 10 ** e
        assert min_eta <= 10 ** e <= max_eta

        min_e, max_e = (np.log10(min_lambda), np.log10(max_lambda))
        e = min_e + (max_e - min_e) * np.random.rand()
        p['lambda'] = 10 ** e

        configs.append(p)
    return configs


def params_search(min_eta, max_eta, min_lambda, max_lambda, silent=False, limit_N=None, combs=100):

    if not silent:
        print("Running parameters search...")

    np.random.seed(42)

    X, Y, y = dataset.load_cifar10(batch='data_batch_1', limit_N=limit_N)
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch', limit_N=limit_N)
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, 30, K]

    default_params = {
        'eta': 0.020,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.9,
        'decay_rate': 0.98,
        'lambda': 0.000001
    }

    configs = _get_configs(default_params, combs, min_eta, max_eta, min_lambda, max_lambda)

    parallel = Parallel(n_jobs=8, backend='multiprocessing', verbose=5)
    results = parallel(delayed(_search_worker)(net_sizes, c, X, Y, X_test, Y_test) for c in configs)

    if not silent:
        print("Parameters search done.")

    return configs, results


def coarse_search(combs=100):
    min_eta, max_eta = (10 ** (-1.85), 10 ** (-1.55))
    min_lambda, max_lambda = (10 ** (-6), 10 ** (-2.2))
    configs, results = params_search(min_eta, max_eta, min_lambda, max_lambda, combs=combs)
    pd.DataFrame(configs).to_pickle(COARSE_SEARCH_CONFIGS_PATH)
    pd.DataFrame(results).to_pickle(COARSE_SEARCH_RESULTS_PATH)


def fine_search(combs=100):
    min_eta, max_eta = (10 ** (-1.85), 10 ** (-1.55))
    min_lambda, max_lambda = (10 ** (-6), 10 ** (-2.2))
    configs, results = params_search(min_eta, max_eta, min_lambda, max_lambda, combs=combs)
    pd.DataFrame(configs).to_pickle(FINE_SEARCH_CONFIGS_PATH)
    pd.DataFrame(results).to_pickle(FINE_SEARCH_RESULTS_PATH)


if __name__ == '__main__':

    if not exists(COARSE_SEARCH_RESULTS_PATH):
        coarse_search(combs=1)
    results_coarse_df = unpickle(COARSE_SEARCH_RESULTS_PATH)
    configs_coarse_df = unpickle(COARSE_SEARCH_CONFIGS_PATH)


    exit(0)

    if not exists(FINE_SEARCH_RESULTS_PATH):
        fine_search(combs=100)
    results_fine_df = unpickle(FINE_SEARCH_RESULTS_PATH)
    configs_fine_df = unpickle(FINE_SEARCH_CONFIGS_PATH)

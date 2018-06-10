#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from os.path import exists
from utilities import unpickle
from model import Net
import dataset


FINE_SEARCH_CONFIGS_PATH = '../configs_fine.pkl'
FINE_SEARCH_RESULTS_PATH = '../results_fine.pkl'
COARSE_SEARCH_CONFIGS_PATH = '../configs_coarse.pkl'
COARSE_SEARCH_RESULTS_PATH = '../results_coarse.pkl'


def _search_worker(net_sizes, config, Ws, bs, X, Y, X_test, Y_test):
    net = Net(net_sizes, config, init_theta=False)
    net.Ws = [W.copy() for W in Ws]
    net.bs = [b.copy() for b in bs]
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
        'lambda': 0.000001,
        'batch_normalize': True
    }

    configs = _get_configs(default_params, combs, min_eta, max_eta, min_lambda, max_lambda)

    net = Net(net_sizes, default_params)
    Ws, bs = net.Ws, net.bs

    parallel = Parallel(n_jobs=8, backend='multiprocessing', verbose=5)
    results = parallel(delayed(_search_worker)(net_sizes, c, Ws, bs, X, Y, X_test, Y_test) for c in configs)

    if not silent:
        print("Parameters search done.")

    return configs, results


def print_top3(configs_df, results_df):
    test_accs = np.array([x[-1] for x in results_df['test_accuracies']])
    top3 = list(np.argsort(test_accs)[-3:])
    top3.reverse()
    print('Top 3 configurations (1st, 2nd, 3rd):\n')
    for i in top3:
        config = configs_df.iloc[i]
        print("Test accuracy: ", test_accs[i], "\n", config, "\n")


def plot_search(configs_df, results_df, fig_path=None, exponents=False):
    lambs = np.array(configs_df['lambda'])
    etas = np.array(configs_df['eta'])
    test_accs = np.array([x[-1] for x in results_df['test_accuracies']])
    plt.figure()

    if exponents:
        lambs = np.log10(lambs)
        etas = np.log10(etas)
        plt.xlabel(r'$log_{10}(\lambda)$')
        plt.ylabel(r'$log_{10}(\eta)$')
    else:
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\eta$')

    plt.scatter(lambs, etas, c=test_accs, marker='^')
    cb = plt.colorbar()
    cb.set_label('test accuracy')
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def coarse_search(combs=100):
    range_e_eta = (-2.00, -1.00)
    range_e_lam = (-6.00, -1.00)
    ranges = tuple(10**x for x in (*range_e_eta, *range_e_lam))
    configs, results = params_search(*ranges, combs=combs)
    pd.DataFrame(configs).to_pickle(COARSE_SEARCH_CONFIGS_PATH)
    pd.DataFrame(results).to_pickle(COARSE_SEARCH_RESULTS_PATH)


def fine_search(combs=100):
    range_e_eta = (-2.00, -1.70)
    range_e_lam = (-6.00, -3.00)
    ranges = tuple(10**x for x in (*range_e_eta, *range_e_lam))
    configs, results = params_search(*ranges, combs=combs)
    pd.DataFrame(configs).to_pickle(FINE_SEARCH_CONFIGS_PATH)
    pd.DataFrame(results).to_pickle(FINE_SEARCH_RESULTS_PATH)


def run_search():

    print('Coarse search:')
    if not exists(COARSE_SEARCH_RESULTS_PATH):
        coarse_search(combs=100)
    results_coarse_df_ = unpickle(COARSE_SEARCH_RESULTS_PATH)
    configs_coarse_df_ = unpickle(COARSE_SEARCH_CONFIGS_PATH)
    print_top3(configs_coarse_df_, results_coarse_df_)
    plot_search(configs_coarse_df_, results_coarse_df_, '../Report/Figs/coarse.eps')

    print('\n\n')

    print('Fine search:')
    if not exists(FINE_SEARCH_RESULTS_PATH):
        fine_search(combs=100)
    results_fine_df_ = unpickle(FINE_SEARCH_RESULTS_PATH)
    configs_fine_df_ = unpickle(FINE_SEARCH_CONFIGS_PATH)
    print_top3(configs_fine_df_, results_fine_df_)
    plot_search(configs_fine_df_, results_fine_df_, '../Report/Figs/fine.eps')


if __name__ == '__main__':
    run_search()

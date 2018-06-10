#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import numpy as np
import matplotlib.pyplot as plt
from model import Net
from utilities import compute_gradients_num, compute_gradients_num_slow, compare_dthetas
import dataset


def test_two_layers(batch_normalize, eta=0.024749):

    np.random.seed(42)
    X, Y, y = dataset.load_multibatch_cifar10()
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, K]
    gd_params = {
        'eta': eta,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.0,
        'decay_rate': 1.00,
        'lambda': 0.000242,
        'batch_normalize': batch_normalize
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

    return r


def test_three_layers(batch_normalize, eta=0.015661):

    np.random.seed(42)
    X, Y, y = dataset.load_multibatch_cifar10()
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, 30, K]

    gd_params = {
        'eta': eta,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.0,
        'decay_rate': 1.0,
        'lambda': 0.000047,
        'batch_normalize': batch_normalize
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

    return r


def test_four_layers(batch_normalize):

    np.random.seed(42)
    X, Y, y = dataset.load_multibatch_cifar10()
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, 30, 10, K]

    gd_params = {
        'eta': 0.03,
        'batch_size': 100,
        'epochs': 20,
        'gamma': 0.9,
        'decay_rate': 0.95,
        'lambda': 0.0,
        'batch_normalize': batch_normalize
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    # costs, test_costs = (r['costs'], r['test_costs'])
    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

    return r


def test_import():

    filepath = './model_epoch_20.pkl'
    net = Net.import_model(filepath)

    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    acc = net.compute_accuracy(X_test, y_test)
    print('test acc', acc)


def test_vs_A2(batch_normalize):

    np.random.seed(42)
    X, Y, y = dataset.load_cifar10(batch='data_batch_1', limit_N=None)
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch', limit_N=None)
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, K]
    gd_params = {
        'eta': 0.024749,
        'batch_size': 100,
        'epochs': 10,
        'gamma': 0.9,
        'decay_rate': 0.80,
        'lambda': 0.000242,
        'batch_normalize': batch_normalize
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

    return r


def test_gradients(batch_normalize):

    np.random.seed(42)
    X, Y, y =  dataset.load_cifar10(batch='data_batch_1', limit_N=100, limit_d=100)
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, 30, 20, K]
    gd_params = {
        'lambda': 0.0,
        'batch_normalize': batch_normalize
    }

    net = Net(net_sizes, gd_params)

    print('\nComputing gradients (analytical methods)...')

    if batch_normalize:
        ss, s_means, s_vars, Hs, P = net._forward_bn(X)
        dtheta = net._backward_bn(X, Y, P, Hs, ss, s_means, s_vars)
    else:
        Hs, P = net._forward(X)
        dtheta = net._backward(X, Y, P, Hs)

    dummy_net = Net(net.network_sizes, gd_params, init_theta=False)
    def dummy_cost_fn(_X, _Y, _W, _b, _lamb):
        dummy_net.Ws = _W
        dummy_net.bs = _b
        return dummy_net.compute_cost(_X, _Y)

    print('Computing gradients (fast numerical method)...')
    dtheta_num = compute_gradients_num(X, Y, net.Ws, net.bs, net.lamb, 1e-5, dummy_cost_fn)

    print('Computing gradients (slow numerical method)...')
    dtheta_num_slow = compute_gradients_num_slow(X, Y, net.Ws, net.bs, net.lamb, 1e-5, dummy_cost_fn)

    print('\nDone\n')

    print('Mean relative errors between numerical methods:\n{}\n'.format(
        compare_dthetas(dtheta_num, dtheta_num_slow)
    ))

    print('Mean relative errors between analytical and slow numerical:\n{}\n'.format(
        compare_dthetas(dtheta, dtheta_num_slow)
    ))

    print('Mean relative errors between analytical and fast numerical:\n{}\n'.format(
        compare_dthetas(dtheta, dtheta_num)
    ))


def plot_two_layers():

    etas = [0.01,0.025,0.04]

    for i, eta in enumerate(etas):

        r = test_two_layers(batch_normalize=False, eta=eta)
        r_bn = test_two_layers(batch_normalize=True, eta=eta)

        test_loss = r['test_losses']
        test_loss_bn = r_bn['test_losses']

        loss = r['losses']
        loss_bn = r_bn['losses']

        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 4.0))

        axes[0].plot(
            np.arange(len(loss)), loss,
            np.arange(len(test_loss)), test_loss
        )

        axes[1].plot(
            np.arange(len(loss_bn)), loss_bn,
            np.arange(len(test_loss_bn)), test_loss_bn
        )

        if i == 0:
            axes[0].set_title(r'Without Batch Normalization')
            axes[1].set_title(r'With Batch Normalization')

        for ax in axes:
            ax.legend(['training loss', 'validation loss'])
            ax.set_ylabel('loss')
            ax.set_ylim(1.22, 2.38)
            if i == 2:
                ax.set_xlabel('epoch')

        plt.subplots_adjust(wspace=0.11, hspace=0.2)
        plt.savefig(f'../Report/Figs/bn_2_layers_{i}.eps', bbox_inches='tight')


def plot_three_layers():

    r = test_three_layers(batch_normalize=False, eta=0.025)
    r_bn = test_three_layers(batch_normalize=True, eta=0.025)

    test_acc = r['test_accuracies']
    test_acc_bn = r_bn['test_accuracies']

    acc = r['accuracies']
    acc_bn = r_bn['accuracies']

    test_loss = r['test_losses']
    test_loss_bn = r_bn['test_losses']

    loss = r['losses']
    loss_bn = r_bn['losses']

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 4.0))

    axes[0].plot(
        np.arange(len(acc)), acc,
        np.arange(len(test_acc)), test_acc
    )
    axes[0].set_title(r'Without Batch Normalization')

    axes[1].plot(
        np.arange(len(acc_bn)), acc_bn,
        np.arange(len(test_acc_bn)), test_acc_bn
    )
    axes[1].set_title(r'With Batch Normalization')

    for ax in axes:
        ax.legend(['training accuracy', 'validation accuracy'])
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')

    plt.subplots_adjust(wspace=0.11, hspace=0.2)
    plt.savefig('../Report/Figs/bn_3_layers.eps', bbox_inches='tight')


    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11, 4.0))

    axes[0].plot(
        np.arange(len(loss)), loss,
        np.arange(len(test_loss)), test_loss
    )

    axes[1].plot(
        np.arange(len(loss_bn)), loss_bn,
        np.arange(len(test_loss_bn)), test_loss_bn
    )

    for ax in axes:
        ax.legend(['training loss', 'validation loss'])
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')

    plt.subplots_adjust(wspace=0.11, hspace=0.2)
    plt.savefig('../Report/Figs/bn_3_layers_loss.eps', bbox_inches='tight')


def plot_results(res, output_path=None):

    f_costs, f_test_costs = (res['costs'], res['test_costs'])
    f_accuracies, f_test_accuracies = (res['accuracies'], res['test_accuracies'])

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(11, 3.8))

    axes[0].plot(np.arange(len(f_costs)), f_costs, np.arange(len(f_costs)), f_test_costs)
    axes[0].legend(['training cost', 'validation cost'])
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('cost')

    axes[1].plot(np.arange(len(f_accuracies)), f_accuracies, np.arange(len(f_test_accuracies)), f_test_accuracies)
    axes[1].legend(['training accuracy', 'validation accuracy'])
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')

    plt.subplots_adjust(wspace=0.18, hspace=0.2)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')

    plt.show()


def test_final_model():

    np.random.seed(42)
    X, Y, y = dataset.load_multibatch_cifar10()
    X_test, Y_test, y_test = dataset.load_cifar10(batch='test_batch')
    K, d = (Y.shape[0], X.shape[0])

    net_sizes = [d, 50, 30, K]

    gd_params = {
        'eta': 0.0169,
        'batch_size': 100,
        'epochs': 20,
        'gamma': 0.6,
        'decay_rate': 0.93,
        'lambda': 5e-5,
        'plateau_guard': 0.0002,
        'batch_normalize': True
    }

    net = Net(net_sizes, gd_params)
    r = net.train(X, Y, X_test, Y_test, silent=False)

    losses, test_losses, accuracies, test_accuracies = (r['losses'], r['test_losses'], r['accuracies'], r['test_accuracies'])

    print("Final accuracy: {}".format(accuracies[-1]))
    print("Final accuracy (test): {}".format(test_accuracies[-1]))

    plot_results(r, '../Report/Figs/final_model.eps')

    return r


if __name__ == "__main__":
    test_final_model()

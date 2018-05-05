#
# KTH Royal Institute of Technology
# DD2424: Deep Learning in Data Science
# Assignment 3
#
# Carlo Rapisarda (carlora@kth.se)
#

import numpy as np
from timeit import default_timer as timer

class Net:

    def __init__(self, network_sizes, descent_params):
        self.network_sizes = network_sizes
        self.n_layers = len(network_sizes)-1
        self.descent_params = descent_params
        self.lamb = descent_params.get('lambda', 0.0)
        self.Ws, self.bs = self._initial_theta()

    def _initial_theta(self):
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

    def _should_batch_normalize(self):
        return self.descent_params.get('batch_normalize', True)

    @staticmethod
    def softmax(s, axis=0):
        exp_s = np.exp(s)
        exp_sum = np.sum(exp_s, axis=axis)
        return exp_s / exp_sum

    def forward(self, X):
        """
        :param X: the training samples
        :return: (Hs, P) where Hs are the outputs of each layer, including the input, excluding the output,
        and P is a matrix with the probabilities for each label for the image in the corresponding column of X
        """

        if self._should_batch_normalize():
            _, _, _, Hs, P = self.forward_bn(X)
            return Hs, P

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

    @staticmethod
    def batch_normalize_s(s):
        mean = np.mean(s, axis=1).reshape(-1,1)
        var = np.var(s, axis=1).reshape(-1,1)
        s_norm = (s - mean) / var
        return s_norm, mean, var

    def forward_bn(self, X):

        Hi = X #.copy()
        si = None
        Hs = []
        s_means = []
        s_vars = []
        ss = []

        for i in range(self.n_layers):

            Hs.append(Hi)
            Wi, bi = self.Ws[i], self.bs[i]

            si = Wi @ Hi + bi
            ss.append(si)

            si, mean, var = self.batch_normalize_s(si)
            s_means.append(mean)
            s_vars.append(var)

            Hi = np.maximum(0.0, si)

        P = self.softmax(ss[-1])
        assert np.isfinite(P).all()
        return ss, s_means, s_vars, Hs, P

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

    @staticmethod
    def batch_normalize_G_fast(G, si, mean, var):

        N = G.shape[1]
        eps = 1e-5 # FIXME: How to set this?

        var_eps = var + eps
        si_zero_mean = si - mean

        dVar_f = -0.5 * np.sum(G * (var_eps**(-3/2.0)) * si_zero_mean, axis=1).reshape(-1,1)
        dMean_f = -np.sum(G * (var_eps**(-1/2.0)), axis=1).reshape(-1,1)

        return G * (var_eps**(-1/2.0)) + (2.0/N * dVar_f * si_zero_mean) + dMean_f/N

    @staticmethod
    def batch_normalize_G(G, si, mean, var):
        assert False, "Deprecated, use batch_normalize_G_fast"

        N = G.shape[1]
        eps = 1e-5 # FIXME: How to set this?
        V = np.diag(var.reshape(-1) + eps)
        V[V == 0.0] = np.inf

        dMean = 0.0
        dVar = 0.0
        for j in range(N):
            dVar = dVar +   G[:,j] @ (V**(-3/2.0)) @ np.diag(si[:,j] - mean.reshape(-1))
            dMean = dMean + G[:,j] @ (V**(-1/2.0))
        dVar = -0.5 * dVar
        dMean = -dMean

        G_norm = np.zeros(G.shape)
        for j in range(N):
            G_norm[:,j] = G[:,j] @ (V**(-1/2.0)) + 2.0/N * dVar @ np.diag(si[:,j] - mean.reshape(-1)) + dMean/N

        return G_norm

    def compute_gradients_fast_bn(self, X, Y, P, Hs, ss, s_means, s_vars):

        N = X.shape[1]
        G = (P - Y)
        grads_W, grads_b = [], []

        for j in range(self.n_layers):

            i = self.n_layers -1 -j # Reversed
            Hi, Wi = Hs[i], self.Ws[i]

            grad_bi = np.mean(G, axis=1).reshape(-1, 1)
            grad_Wi = (G @ Hi.T) / N + (2 * self.lamb * Wi)

            assert np.isfinite(grad_bi).all()
            assert np.isfinite(grad_Wi).all()

            grads_b.append(grad_bi)
            grads_W.append(grad_Wi)

            G = (G.T @ Wi).T
            G[Hi <= 0] = 0.0

            if i > 0:
                si, mean, var = ss[i-1], s_means[i-1], s_vars[i-1]
                assert si.shape == G.shape
                G = self.batch_normalize_G_fast(G, si, mean, var)

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
        plateau_guard = params.get('plateau_guard', None)
        batch_normalize = self._should_batch_normalize()

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
                if batch_normalize:
                    ss, s_means, s_vars, Hs, P = self.forward_bn(X_batch)
                    grads_W, grads_b = self.compute_gradients_fast_bn(X_batch, Y_batch, P, Hs, ss, s_means, s_vars)
                else:
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

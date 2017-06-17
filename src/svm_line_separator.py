import numpy as np
from numpy.random import seed, randn, RandomState
import matplotlib
from matplotlib import pyplot as plt


class LineSeparator:
    def __init__(self, X, y, learning_rate=0.00001, regularize=100.):
        """
        X: [[x0[0], x1[0], ...], [x0[1], x1[1], ...]]
        y: [y0, y1, ...]
        """
        self._X = X
        self._y = y
        self._data_num = y.shape[0]

        self._thetas = np.zeros(self._data_num)
        self._bias = 0
        self._learning_rate = learning_rate
        self._lambda = regularize

    def train(self, epochs=150):
        for epoch in range(epochs):
            # Show log
            if epoch % 100 == 0:
                self._pretty_print(epoch, self.omega)

            # Updata thetas
            new_thetas = np.copy(self._thetas)
            for k, _ in enumerate(self._thetas):
                update = 0
                for j in range(self._data_num):
                    update += self._subdiff_theta(j, k)
                new_thetas[k] -= self._learning_rate * update
            new_thetas -= self._learning_rate * self._lambda * self._X.T @ self._X @ self._thetas
            self._thetas = new_thetas

            # Updata b
            for j in range(self._data_num):
                self._bias -= self._learning_rate * self._subdiff_b(j)

        w = self.omega
        self._pretty_print(None, w, bias=self._bias)
        return w, self._bias

    @property
    def omega(self):
        """
        omega = sum theta_j psi(x_j)
        """
        return self._X @ self._thetas

    @property
    def b(self):
        return self._bias

    def _pretty_print(self, epoch, omegas, bias=None):
        if epoch is None:
            log = "Finally:\n"
        else:
            log = "Epoch %d:\n" % epoch

        for i, omega in enumerate(omegas):
            log += "   w%d: %e\n" % (i + 1, omega)

        if bias is not None:
            log += "   b: %f\n" % bias

        print(log)

    def _f(self, x):
        """
        x: [x[0], x[1]]
        Returns: f_theta(x) = w x + b
        """
        return x.T @ self._X @ self._thetas + self._bias

    def _subdiff_theta(self, i, k):
        """
        i-th subdifferentiation by k_th theta
        """
        if 1 - self._y[i] * self._f(self._X[:, i]) > 0:
            return - self._y[i] * self._X[:, k] @ self._X[:, i]
        else:
            return 0

    def _subdiff_b(self, i):
        """
        i-th subdifferentiate by b
        """
        if 1 - self._y[i] * self._f(self._X[:, i]) > 0:
            return - self._y[i]
        else:
            return 0


def prepare_data(n, pos_rate=0.9, random_state=None):
    if type(random_state) is RandomState:
        rs = random_state
    else:
        rs = RandomState(42)
    pos_num = int(n * pos_rate)
    neg_num = n - pos_num
    X = np.array([np.concatenate((rs.randn(pos_num) - 2, rs.randn(neg_num) + 2)), 2 * rs.randn(n)])
    y = np.concatenate((np.ones(pos_num), - np.ones(neg_num)))
    return X, y


def main():
    rs = RandomState(42)
    train_X, train_y = prepare_data(100, 0.9, rs)
    X, y = prepare_data(100, 0.1, rs)

    train_line_sep = LineSeparator(train_X, train_y)
    test_line_sep = LineSeparator(X, y)

    tr_w, tr_b = train_line_sep.train(500)
    w, b = test_line_sep.train(500)
    line_points = np.linspace(-5, 5)

    fig = plt.figure(figsize=(10, 10))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax1.plot(train_X[0, train_y == 1], train_X[1, train_y == 1], 'bo')
    ax1.plot(train_X[0, train_y == -1], train_X[1, train_y == -1], 'rx')
    ax1.plot(line_points, - (tr_b + line_points * tr_w[0]) / tr_w[1], ':')
    ax1.plot(line_points, - (b + line_points * w[0]) / w[1], 'g')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-10, 10)

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2.plot(X[0, y == 1], X[1, y == 1], 'bo')
    ax2.plot(X[0, y == -1], X[1, y == -1], 'rx')
    ax2.plot(line_points, - (tr_b + line_points * tr_w[0]) / tr_w[1], ':')
    ax2.plot(line_points, - (b + line_points * w[0]) / w[1], 'g')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-10, 10)

    fig.show()

import numpy as np
from numpy.random import seed, randn, RandomState
import matplotlib
from matplotlib import pyplot as plt


class LineSeparator:
    def __init__(self, data, learning_rate=0.00001, regularize=100.):
        """
        X: [[x0[0], x1[0], ...], [x0[1], x1[1], ...]]
        y: [y0, y1, ...]
        """
        self._data = data

        self._thetas = np.zeros(self._data.data_num)
        self._bias = 0
        self._learning_rate = learning_rate
        self._lambda = regularize

    def train(self, compensate_class_balance=False, epochs=150):
        for epoch in range(epochs):
            # Show log
            if epoch % 200 == 0:
                self._pretty_print(epoch, self.omega)

            # Updata thetas
            new_thetas = np.copy(self._thetas)
            for k, _ in enumerate(self._thetas):
                update = 0
                for j in range(self._data.data_num):
                    update += self._subdiff_theta(j, k, compensate_class_balance)
                new_thetas[k] -= self._learning_rate * update
            new_thetas -= self._learning_rate * self._lambda * self._data.train_X.T @ self._data.train_X @ self._thetas
            self._thetas = new_thetas

            # Updata b
            for j in range(self._data.data_num):
                self._bias -= self._learning_rate * self._subdiff_b(j, compensate_class_balance)

        w = self.omega
        self._pretty_print(None, w, bias=self._bias)
        return w, self._bias

    @property
    def omega(self):
        """
        omega = sum theta_j psi(x_j)
        """
        return self._data.train_X @ self._thetas

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
        return x.T @ self._data.train_X @ self._thetas + self._bias

    def _subdiff_theta(self, i, k, compensate_class_balance=False):
        """
        i-th subdifferentiation by k_th theta
        """
        if 1 - self._data.train_y[i] * self._f(self._data.train_X[:, i]) > 0:
            if compensate_class_balance:
                return - self._data.train_y[i] * self._data.train_X[:, k] @ self._data.train_X[:, i] * self._data.importance(self._data.train_y[i])
            else:
                return - self._data.train_y[i] * self._data.train_X[:, k] @ self._data.train_X[:, i]
        else:
            return 0

    def _subdiff_b(self, i, compensate_class_balance=False):
        """
        i-th subdifferentiate by b
        """
        if 1 - self._data.train_y[i] * self._f(self._data.train_X[:, i]) > 0:
            if compensate_class_balance:
                return - self._data.train_y[i] * self._data.importance(self._data.train_y[i])
            else:
                return - self._data.train_y[i]
        else:
            return 0


class ClassBalanceEstimater:
    def __init__(self, train_X, train_y, test_X):
        self._X = train_X
        self._y = train_y
        self._test_X = test_X
        self._pi = self._calc_pi()

    @property
    def train_X(self):
        return self._X

    @property
    def train_y(self):
        return self._y

    @property
    def test_X(self):
        return self._test_X

    @property
    def data_num(self):
        return self._X.shape[1]

    def importance(self, label):
        if label == 1:
            return self._pi
        else:
            return 1 - self._pi

    def _calc_pi(self):
        numerator = self._A(1, -1) - self._A(-1, -1) - self._b(1) + self._b(-1)
        denominator = 2 * self._A(1, -1) - self._A(1, 1) - self._A(-1, -1)
        return numerator / denominator

    def _tile_square(self, X, cols):
        """
        Align X^2 colum-wise
        X: [[x0[0], x1[0], ...], [x0[1], x1[1], ...]]
        Returns: [[x0^2, x0^2, ...], [x1^2, x1^2, ...], ...]
        """
        sq = np.square(X[0, :]) + np.square(X[1, :])
        sq = sq.reshape([X.shape[1], 1])
        return np.tile(sq, cols)

    def _dist_matrix(self, X0, X1):
        """
        X: [[x0[0], x1[0], ...], [x0[1], x1[1], ...]]
        Returns: [[||X0_0 - X1_0||, ||X0_0 - X1_1||, ...], ...]
        """
        X0_sq = self._tile_square(X0, X1.shape[1])
        X1_sq = self._tile_square(X1, X0.shape[1])
        return np.sqrt(X0_sq + X1_sq.T - 2 * X0.T @ X1)

    def _A(self, y0, y1):
        X0 = self._X[:, self._y == y0]
        X1 = self._X[:, self._y == y1]
        n0 = X0.shape[1]
        n1 = X1.shape[1]
        return np.sum(self._dist_matrix(X0, X1)) / (n0 * n1)

    def _b(self, y):
        X = self._X[:, self._y == y]
        n = X.shape[1]
        n_test = self._test_X.shape[1]
        return np.sum(self._dist_matrix(self._test_X, X)) / (n * n_test)


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
    rs = RandomState(13)
    train_X, train_y = prepare_data(100, 0.9, rs)
    X, y = prepare_data(100, 0.1, rs)

    class_balance_estimater = ClassBalanceEstimater(train_X, train_y, X)
    line_sep = LineSeparator(class_balance_estimater)
    # test_line_sep = LineSeparator(X, y)

    tr_w, tr_b = line_sep.train(epochs=1000)
    real_w, real_b = line_sep.train(compensate_class_balance=True, epochs=1000)
    # w, b = test_line_sep.train(500)
    line_points = np.linspace(-5, 5)

    fig = plt.figure(figsize=(10, 10))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax1.plot(train_X[0, train_y == 1], train_X[1, train_y == 1], 'bo')
    ax1.plot(train_X[0, train_y == -1], train_X[1, train_y == -1], 'rx')
    ax1.plot(line_points, - (tr_b + line_points * tr_w[0]) / tr_w[1], 'k:')
    ax1.plot(line_points, - (real_b + line_points * real_w[0]) / real_w[1], 'g')
    # ax1.plot(line_points, - (b + line_points * w[0]) / w[1], 'g')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-10, 10)

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2.plot(X[0, y == 1], X[1, y == 1], 'bo')
    ax2.plot(X[0, y == -1], X[1, y == -1], 'rx')
    ax2.plot(line_points, - (tr_b + line_points * tr_w[0]) / tr_w[1], 'k:')
    ax2.plot(line_points, - (real_b + line_points * real_w[0]) / real_w[1], 'g')
    # ax2.plot(line_points, - (b + line_points * w[0]) / w[1], 'g')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-10, 10)

    fig.show()

import numpy as np
from numpy.random import rand, randn, RandomState
from matplotlib import pyplot as plt
import matplotlib


def prepare_data(n):
    X0 = np.array([2 * randn(n), randn(n)])
    X1 = np.array([2 * randn(n), 2 * np.round(rand(n)) - 1 + randn(n) / 3])
    return X0, X1


def main():
    X0, X1 = prepare_data(200)
    d0, d1 = Data(X0), Data(X1)
    lpp_0 = LPP(d0)
    lpp_1 = LPP(d1)
    zeta0 = lpp_0.calculate_T(1)[0]
    zeta1 = lpp_1.calculate_T(1)[0]

    line_points = np.linspace(-6, 6, 10)

    fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax1.plot(X0[0, :], X0[1, :], 'bo')
    ax1.plot(line_points, line_points * zeta0[1] / zeta0[0], 'g')
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-4, 4)

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2.plot(X1[0, :], X1[1, :], 'bo')
    ax2.plot(line_points, line_points * zeta1[1] / zeta1[0], 'g')
    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-4, 4)

    fig.show()


class Data:
    def __init__(self, X):
        """
        X: [[x0[0], x1[0], ...], [x0[1], x1[1], ...]]
        y: [y0, y1, ...]
        """
        self._X = X

    @property
    def X(self):
        return self._X

    def kernel_matrix(self, h):
        return np.exp(- self._square_dist() / (2 * h**2))

    def _square_dist(self):
        sqr = np.square(self._X[0]) + np.square(self._X[1])
        sqr_m = np.tile(sqr, (sqr.shape[0], 1))
        return sqr_m + sqr_m.T - 2 * self._X.T @ self._X


class LPP:
    def __init__(self, data):
        self._data = data

    def _weight_matrix(self):
        """
        Distance based matrix
        h = 0.5 is a magic number
        """
        return self._data.kernel_matrix(0.5)

    def _D_matrix(self):
        return np.diag(np.sum(self._weight_matrix(), axis=0))

    def _laplacian_matrix(self):
        return self._D_matrix() - self._weight_matrix()

    def calculate_T(self, dim: int):
        """
        dim: Number of eighen vectors.
        Return: T = [e_vec_0, e_vec_1, ..., e_vec_dim]
        """
        A = self._data.X @ self._laplacian_matrix() @ self._data.X.T
        B = self._data.X @ self._D_matrix() @ self._data.X.T
        B_half_inv = np.linalg.inv(self._matrix_root(B))
        C = B_half_inv @ A @ B_half_inv
        e_val, e_vec = np.linalg.eigh(C)
        T = np.array([e_vec[:, i] for i in np.argsort(e_val)])
        return T[:dim]

    def _matrix_root(self, A):
        """
        Calculate X s.t. XX = A
        A should be a positive definite matrix.
        """
        e_val, e_vec = np.linalg.eigh(A)
        sqrt_e_val = np.sqrt(e_val)
        return np.sum([v * e_vec[:, i][:, np.newaxis] @ e_vec[:, i][np.newaxis, :]
                       for i, v in enumerate(sqrt_e_val)], axis=0)

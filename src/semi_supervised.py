import numpy as np
from numpy.random import seed, randn


def create_data(n):
    points = np.linspace(0, np.pi, n // 2)
    u = - 10 * np.concatenate([np.cos(points) + 0.5, np.cos(points) - 0.5]).reshape(n, 1) + randn(n, 1)
    v = 10 * np.concatenate([np.sin(points), - np.sin(points)]).reshape(n, 1) + randn(n, 1)
    X = np.array([u, v]).reshape(2, n).transpose((1, 0))
    y = np.zeros((n, 1))
    y[0] = - 1
    y[-1] = 1
    return X, y


def draw_contour(axis_range=(-20, 20), granularity=100, z_func=lambda x, y: x + y):
    # You have to plt.show() after this function
    axis = np.linspace(*axis_range, granularity)
    X, Y = np.meshgrid(axis, axis)
    Z = z_func(X, Y)
    plt.contourf(X, Y, Z)


def kernel_matrix(X, rows=None, h=1.):
    """
    X: [(x0[0], x0[1]), (x1[0], x1[1]), ...]
    """
    if rows is None:
        return np.exp(- dist_matrix(X) / (2 * h**2))
    else:
        return np.exp(- dist_matrix(X)[:rows] / (2 * h**2))


def kern(x, train_X, h=1.):
    """
    x: [(x0[0], x0[1]), (x1[0], x1[1])]
    X: [(x0[0], x0[1]), (x1[0], x1[1]), ...]
    """
    x_sq = tile_square(x, train_X.shape[0])
    X_sq = tile_square(train_X, x.shape[0]).transpose((1, 0))
    dist = x_sq + X_sq - 2 * x @ train_X.T
    return np.exp(- dist / (2 * h**2))


def tile_square(X, cols):
    """
    Align X^2 colum-wise
    X: [(x0[0], x0[1]), (x1[0], x1[1]), ...]
    """
    sq = np.square(X[:, 0]) + np.square(X[:, 1])
    sq = sq.reshape([X.shape[0], 1])
    return np.tile(sq, cols)


def dist_matrix(X):
    """
    X: [(x0[0], x0[1]), (x1[0], x1[1]), ...]
    Returns: [[(x0 - x0)^2, (x0 - x1)^2, ...], ...]
    """
    sq = tile_square(X, X.shape[0])
    return sq + sq.T - 2 * X @ X.T


def weight_matrix(X, k=10):
    """
    X: [(x0[0], x0[1]), (x1[0], x1[1]), ...]
    k: k nearest neighbor
    """
    n = X.shape[0]
    W = np.zeros((n, n))
    dist = dist_matrix(X)
    for i in range(n):
        for j in range(i + 1, n):
            if is_mutually_k_nearest(dist, i, j, k):
                W[i, j] = 1.
                W[j, i] = 1.
    return W


def is_mutually_k_nearest(dist_matrix, i: int, j: int, k: int):
    """
    dist_matrix: [[d(x0, x0), (x0, x1), ...], [d(x1, x0), (x1, x1), ...], ...]
    """
    # Is j k-nearest-neighbour of i ?
    dist_from_i = dist_matrix[i]
    k_nearest_idxes = np.argsort(dist_from_i)[:k + 1]
    if j not in k_nearest_idxes:
        return False
    # Is i k-nearest-neighbour of j ?
    dist_from_j = dist_matrix[j]
    k_nearest_idxes = np.argsort(dist_from_j)[:k + 1]
    if i not in k_nearest_idxes:
        return False
    return True


def laplacian_matrix(X, k=10):
    W = weight_matrix(X, k)
    D = np.diag(np.sum(W, axis=1))
    return D - W


def estimate_theta(labeled_X, labels, unlabeled_X, k=2, lamb=1., nu=1.):
    """
    X: [(x0[0], x0[1]), (x1[0], x1[1]), ...]
    """
    X = np.concatenate((labeled_X, unlabeled_X))
    K = kernel_matrix(X)
    K_l = kernel_matrix(X, labeled_X.shape[0])
    L = laplacian_matrix(X, k)

    Q = K_l.T @ K_l + lamb * np.eye(X.shape[0]) + 2 * nu * K.T @ L @ K
    return np.linalg.inv(Q) @ K_l.T @ labels


def generate_classifier(theta, train_X):
    def classifier(X0, X1):
        """
        X0: [[p0, p1, ...], [...], ...]
        X1: [[q0, q1, ...], [...], ...]
        """
        X = np.array([X0, X1]).transpose((1, 2, 0)).reshape(-1, 2)
        kernels = kern(X, train_X)
        signs = np.array(np.sign(kernels @ theta))
        reshaped = np.array(np.split(signs.reshape(-1), X0.shape[0]))
        return reshaped
    return classifier


def main():
    seed(42)
    X, y = create_data(200)
    labeled_X = X[(y != 0).reshape(-1)]
    unlabeled_X = X[(y == 0).reshape(-1)]
    labels = y[(y != 0).reshape(-1)]
    t = estimate_theta(labeled_X, labels, unlabeled_X, 4)
    classifier = generate_classifier(t, X)

    draw_contour(granularity=500, z_func=classifier)
    plt.plot(X[(y == 1).reshape(-1), 0], [X[(y == 1).reshape(-1), 1]], 'bo')
    plt.plot(X[(y == -1).reshape(-1), 0], [X[(y == -1).reshape(-1), 1]], 'rx')
    plt.plot(X[(y == 0).reshape(-1), 0], X[(y == 0).reshape(-1), 1], '.')
    plt.show()

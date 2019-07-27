import numpy as np


class BaseStep:
    def __init__(self, name):
        self.name = name
        self.params = dict()

    def __str__(self):
        return f'Pipeline step \'{self.name}\''

    def set_param(self, key, param):
        self.params[key] = param

    def set_params(self, params):
        for key in params:
            self.set_param(key, params[key])

    def filter(self, X, y, w):
        accept_all = np.array([True, ] * len(y))

        return accept_all

    def fit(self, X, y, w):
        pass

    def transform(self, X, y, w):
        return X, y, w


class Standardize(BaseStep):
    def __init__(self, name='Standardize'):
        super(Standardize, self).__init__(name)
        self.mean = None
        self.std = None

    @staticmethod
    def avg_and_std(X, w):
        n, _ = X.shape
        avg = np.average(X, axis=0, weights=w)
        var = np.average((X - avg[np.newaxis, :]) ** 2, axis=0, weights=w) * n / (n - 1)

        return avg, np.sqrt(var)

    def fit(self, X, y, w):
        self.mean, self.std = Standardize.avg_and_std(X, w)

    def transform(self, X, y, w):
        assert self.mean is not None
        assert self.std is not None

        return (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :], y, w


class PCA(BaseStep):
    def __init__(self, name='PCA', ignore=None, standardize=True):
        super(PCA, self).__init__(name)
        self.R = None
        self.params['ignore'] = ignore
        self.params['standardize'] = standardize
        self.ignore = ignore
        self.mean = None
        self.std = None

    def fit(self, X, y, w):
        to_list = lambda x: x if hasattr(x, '__iter__') else [x, ]
        ignore = to_list(self.params['ignore']) if 'ignore' in self.params else []

        sel = np.array([True, ] * len(y))
        for label in ignore:
            sel &= (y != label)

        cov = np.cov(X[sel, :].T, aweights=w[sel])
        _, _, RT = np.linalg.svd(cov)
        self.R = RT.T

        self.mean, self.std = Standardize.avg_and_std(X @ self.R, w)

    def transform(self, X, y, w):
        assert self.R is not None
        assert X.shape[1] == self.R.shape[0]
        assert self.mean is not None
        assert self.std is not None
        assert len(self.mean) == X.shape[1]
        assert len(self.std) == X.shape[1]

        Y = X @ self.R

        if self.params['standardize']:
            return (Y - self.mean[np.newaxis, :]) / self.std[np.newaxis, :], y, w
        else:
            return Y, y, w


class BinaryOverlapPCA(PCA):
    def __init__(self, name='BinaryPCA', bin_edges=np.linspace(-5, 5, 101)):
        super(BinaryOverlapPCA, self).__init__(name=name, ignore=None, standardize=True)
        self.bin_edges = bin_edges

    @staticmethod
    def overlap(x1, w1, x2, w2, bin_edges):
        h1, _ = np.histogram(x1 * w1, bin_edges)
        h2, _ = np.histogram(x2 * w2, bin_edges)
        dx = (bin_edges - np.roll(bin_edges, 1))[1:]

        return sum(h1 * h2 * dx)

    def fit(self, X, y, w):
        labels = np.unique(y)
        assert 0 < len(labels) <= 2

        super(BinaryOverlapPCA, self).fit(X, y, w)
        assert self.R is not None

        sel = (y == labels[0])
        X1 = X[sel, :] @ self.R
        w1 = w[sel]
        w2 = w[~sel]
        X2 = X[~sel, :] @ self.R

        assert X1.shape == X2.shape
        _, n = X1.shape
        overlaps = [self.overlap(x1=X1[:, i], w1=w1, x2=X2[:, i], w2=w2, bin_edges=self.bin_edges) for i in range(n)]

        idx = np.argsort(overlaps)
        self.R = self.R[idx, :]

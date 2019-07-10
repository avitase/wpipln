import numpy as np


class BaseStep:
    def __init__(self, name):
        self.name = name
        self.params = dict()

    def __str__(self):
        return f'Pipeline step \'{self.name}\''

    def set_params(self, params):
        self.params = params

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
    def __init__(self, name='PCA', ignore=None):
        super(PCA, self).__init__(name)
        self.R = None
        self.params['ignore'] = ignore
        self.ignore = ignore

    def fit(self, X, y, w):
        to_list = lambda x: x if hasattr(x, '__iter__') else [x, ]
        ignore = to_list(self.params['ignore']) if 'ignore' in self.params else []

        sel = np.array([True, ] * len(y))
        for label in ignore:
            sel &= (y != label)

        cov = np.cov(X[sel, :].T, aweights=w[sel])
        _, _, RT = np.linalg.svd(cov)
        self.R = RT.T

    def transform(self, X, y, w):
        assert self.R is not None

        return X @ self.R, y, w

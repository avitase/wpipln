import numpy as np


class BaseStep:
    def __init__(self, name):
        self.name = name
        self.params = dict()

    def __str__(self):
        return f'Pipeline step \'{self.name}\''

    def set_params(self, params):
        self.params = params

    def fit(self, X, y, w):
        pass

    def transform(self, X, y, w):
        return X, y, w


class Standardize(BaseStep):
    def __init__(self, name='Standardize'):
        super(Standardize, self).__init__(name)
        self.mean = None
        self.std = None

    def fit(self, X, y, w):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X, y, w):
        assert self.mean is not None
        assert self.std is not None

        _, m = X.shape
        for i in range(m):
            X[:, i] = (X[:, i] - self.mean[i]) / self.std[i]

        return X, y, w


class PCA(BaseStep):
    def __init__(self, name='PCA', n_max=None):
        super(PCA, self).__init__(name)
        self.params['n_max'] = n_max
        self.V = None

    def fit(self, X, y, w):
        n, _ = X.shape
        n_max = self.params['n_max'] if 'n_max' in self.params else n
        if n_max == -1 or n_max is None:
            n_max = n

        assert n_max > 0, f'n_max is {n_max} but should be positive'

        _, _, V = np.linalg.svd(X[:n_max, :])
        self.V = V

    def transform(self, X, y, w):
        assert self.V is not None
        assert all(np.abs(X.mean(axis=0)) < 1e-5)

        return X.dot(self.V.T), y, w

import numpy as np

from .BaseStep import BaseStep
from .helper import average, standard_deviation


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

        Y = X @ self.R
        self.mean = average(Y, w)
        self.std = standard_deviation(Y, w)
        self.is_fitted = True

    def transform(self, X, y, w):
        assert self.is_fitted
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

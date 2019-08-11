import numpy as np
from scipy.stats import wasserstein_distance

from .PCA import PCA


class BinaryWPCA(PCA):
    def __init__(self, name='BinaryWPCA'):
        super(BinaryWPCA, self).__init__(name=name, ignore=None, standardize=True)

    def fit(self, X, y, w):
        labels = np.unique(y)
        assert 0 < len(labels) <= 2

        super(BinaryWPCA, self).fit(X, y, w)
        assert self.R is not None

        sel = (y == labels[0])
        X1 = X[sel, :] @ self.R
        w1 = w[sel]
        w2 = w[~sel]
        X2 = X[~sel, :] @ self.R

        assert X1.shape == X2.shape
        _, n = X1.shape
        distances = [wasserstein_distance(u_values=X1[:, i],
                                          u_weights=w1,
                                          v_values=X2[:, i],
                                          v_weights=w2) for i in range(n)]

        idx = np.flip(np.argsort(distances))
        self.R = self.R[:, idx]
        self.is_fitted = True

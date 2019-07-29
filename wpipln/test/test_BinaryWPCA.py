import unittest

import numpy as np
from scipy.stats import wasserstein_distance

from wpipln.pipelines import Pipeline
from wpipln.steps import BinaryWPCA


class TestBinaryWPCA(unittest.TestCase):
    def test_ordering(self):
        pipeline = Pipeline()
        pipeline.add_step('wpca', BinaryWPCA())

        mean1 = [.5, -.5]
        mean2 = [-.5, .5]
        cov = [[1., .9], [.9, 1.]]
        X1 = np.random.multivariate_normal(mean1, cov, 10000)
        X2 = np.random.multivariate_normal(mean2, cov, 10000)

        X = np.vstack((X1, X2))
        y = np.append(np.zeros(10000), np.ones(10000))
        w = np.ones(20000)
        pipeline.fit(X, y, w)

        Xt, _, _ = pipeline.transform(X, y, w)
        is_diag = lambda X: np.allclose(X - np.diag(np.diagonal(X)), np.zeros(X.shape))
        self.assertTrue(is_diag(np.cov(Xt.T)))

        sel1 = (y == 0)
        sel2 = (y == 1)

        distance = lambda x1, w1, x2, w2: wasserstein_distance(u_values=x1,
                                                               v_values=x2,
                                                               u_weights=w1,
                                                               v_weights=w2)

        dist1 = distance(x1=X[sel1, 0], w1=w[sel1], x2=Xt[sel2, 0], w2=w[sel2])
        dist2 = distance(x1=X[sel1, 1], w1=w[sel1], x2=Xt[sel2, 1], w2=w[sel2])

        self.assertLess(dist1, dist2)

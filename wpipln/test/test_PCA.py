import unittest

import numpy as np

from wpipln.pipelines import Pipeline
from wpipln.steps import PCA


class TestPCA(unittest.TestCase):
    def test_PCA(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(standardize=False))

        X = np.array([[1., 2.],
                      [2., 4.],
                      [3., 6.]])
        w = np.array([1., 1., 1.])
        y = np.zeros(3)

        abspcc = lambda a, b: abs(np.corrcoef(a, b)[0][1])
        self.assertGreater(abspcc(X[:, 0], X[:, 1]), .99)

        pipeline.fit(X, y, w)

        pca = pipeline.get_step('pca')
        R = pca.R
        self.assertTrue(np.allclose(R.dot(R.T), np.identity(R.shape[0])))

        Xt, yt, wt = pipeline.transform(X, y, w)
        self.assertTrue(np.allclose(y, yt))
        self.assertTrue(np.allclose(w, wt))

        cov = np.cov(Xt.T)
        self.assertAlmostEqual(cov[0, 0], 5.)
        self.assertAlmostEqual(cov[1, 0], 0.)
        self.assertAlmostEqual(cov[0, 1], 0.)
        self.assertAlmostEqual(cov[1, 1], 0.)

    def test_PCA_ignore(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(ignore=[1, 2], standardize=False))

        X = np.array([[1., 2.],
                      [2., 4.],
                      [3., 6.],
                      [4., 8.],
                      [5., 10.]])
        w = np.array([1., 1., 1., 1., 1.])
        y = np.array([0, 0, 0, 1, 2])

        pipeline.fit(X, y, w)

        Xt, _, _ = pipeline.transform(X, y, w)
        cov = np.cov(Xt[(y != 1) & (y != 2)].T)
        self.assertAlmostEqual(cov[0, 0], 5.)
        self.assertAlmostEqual(cov[1, 0], 0.)
        self.assertAlmostEqual(cov[0, 1], 0.)
        self.assertAlmostEqual(cov[1, 1], 0.)

    def test_standardize(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(standardize=True))

        X = np.array([[1., 2.],
                      [2., 4.],
                      [3., 6.],
                      [4., 8.],
                      [5., 10.]])
        w = np.array([1., 1., 1., 1., 1.])
        y = np.array([0, 0, 0, 1, 2])

        pipeline.fit(X, y, w)
        Xt, _, _ = pipeline.transform(X, y, w)
        for mean, std in zip(Xt.mean(axis=0), Xt.std(axis=0, ddof=1)):
            self.assertAlmostEqual(mean, 0.)
            self.assertAlmostEqual(std, 1.)

    def test_weighting(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(standardize=False))

        X1 = np.array([[1., 2.],
                       [2., 4.],
                       [3., 6.],
                       [4., 8.],
                       [5., 10.]])
        w1 = np.array([0., 1., 2., 1., 1.])
        y1 = np.array([0, 0, 0, 1, 2])

        X2 = np.array([[2., 4.],
                       [3., 6.],
                       [3., 6.],
                       [4., 8.],
                       [5., 10.]])
        w2 = np.array([0., 1., 2., 1., 1.])
        y2 = np.array([0, 0, 0, 1, 2])

        pipeline.fit(X1, y1, w1)
        Xt1, _, _ = pipeline.transform(X1, y1, w1)
        means1 = Xt1.mean(axis=0)
        stds1 = Xt1.std(axis=0, ddof=1)

        pipeline.fit(X2, y2, w2)
        Xt2, _, _ = pipeline.transform(X2, y2, w2)
        means2 = Xt1.mean(axis=0)
        stds2 = Xt1.std(axis=0, ddof=1)

        for mean1, mean2 in zip(means1, means2):
            self.assertAlmostEqual(mean1, mean2)

        for std1, std2 in zip(stds1, stds2):
            self.assertAlmostEqual(std1, std2)

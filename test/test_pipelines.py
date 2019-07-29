import unittest

import numpy as np

import test.pipelines
import test.steps
from test.helper import fit, transform, mat_eq
from wpipln.pipelines import Pipeline, BalancedPipeline
from wpipln.steps import BaseStep, Standardize, PCA, BinaryOverlapPCA


class TestPipeline(unittest.TestCase):
    def test_unit_step(self):
        pipeline = Pipeline()
        pipeline.add_step('unit', BaseStep('unit'), [2, 4])
        self.assertTrue(pipeline.has_step('unit'))

        X = np.arange(50).reshape((10, 5))
        y = np.arange(10)
        w = np.arange(10) + 10

        fit(pipeline, X, y, w)

        Xt, yt, wt = transform(pipeline, X, y, w)
        self.assertTrue(mat_eq(X, Xt))
        self.assertTrue(mat_eq(y, yt))
        self.assertTrue(mat_eq(w, wt))

    def test_zero_step(self):
        pipeline = Pipeline()
        indices = [2, 4]
        pipeline.add_step('zero', test.steps.ZeroXStep(), indices)
        self.assertTrue(pipeline.has_step('zero'))

        X = np.arange(50).reshape((10, 5))
        y = np.arange(10)
        w = np.arange(10)

        fit(pipeline, X, y, w)
        Xt, _, _ = transform(pipeline, X, y, w)

        X[:, indices] = 0
        self.assertTrue(mat_eq(X, Xt))

    def test_indices_wildcard(self):
        pipeline = Pipeline()
        pipeline.add_step('zero', test.steps.ZeroXStep(), '*')
        self.assertTrue(pipeline.has_step('zero'))

        X = np.arange(50).reshape((10, 5))
        y = np.arange(10)
        w = np.arange(10)

        fit(pipeline, X, y, w)
        Xt, _, _ = transform(pipeline, X, y, w)

        X[:, :] = 0
        self.assertTrue(mat_eq(X, Xt))

    def test_fit_step(self):
        pipeline = Pipeline()
        indices = [2, 4]
        pipeline.add_step('center', test.steps.CenterStep(), indices)

        X = np.random.rand(50).reshape((5, 10))
        X[:, indices[0]] = np.arange(5).astype(float)
        X[:, indices[1]] = np.arange(5).astype(float) * 2
        mean = np.mean(X[:, indices], axis=0)
        self.assertAlmostEqual(mean[0], 2.)
        self.assertAlmostEqual(mean[1], 4.)

        y = np.arange(5)
        w = np.arange(5)

        fit(pipeline, X, y, w)

        X = np.random.rand(50).reshape((5, 10))
        Xt, _, _ = transform(pipeline, X, y, w)

        X[:, indices[0]] -= mean[0]
        X[:, indices[1]] -= mean[1]

        self.assertTrue(mat_eq(X, Xt))

    def test_compose(self):
        pipeline = Pipeline()
        indices = [2, 4]
        pipeline \
            .add_step('center', test.steps.CenterStep(), indices) \
            .add_step('stdscale', test.steps.StdScaleStep(), indices)

        X = np.random.rand(70).reshape((7, 10))
        X[:, indices[0]] = np.arange(7).astype(float)
        X[:, indices[1]] = np.arange(7).astype(float) * 2

        mean = np.mean(X[:, indices], axis=0)
        self.assertAlmostEqual(mean[0], 3.)
        self.assertAlmostEqual(mean[1], 6.)

        std = np.std(X[:, indices], axis=0)
        self.assertAlmostEqual(std[0], 2.)
        self.assertAlmostEqual(std[1], 4.)

        y = np.arange(7)
        w = np.arange(7)

        fit(pipeline, X, y, w)
        Xt1, _, _ = transform(pipeline, X, y, w, last_step='center')
        Xt2, _, _ = transform(pipeline, X, y, w)

        X_exp1 = np.array(X, copy=True)
        X_exp1[:, indices[0]] -= mean[0]
        X_exp1[:, indices[1]] -= mean[1]

        X_exp2 = np.array(X_exp1, copy=True)
        X_exp2[:, indices[0]] /= std[0]
        X_exp2[:, indices[1]] /= std[1]

        self.assertTrue(mat_eq(X_exp1, Xt1))
        self.assertTrue(mat_eq(X_exp2, Xt2))

    def test_inner_pipeline(self):
        pipeline = test.pipelines.DiscardPipeline(-1, name='outer_pipln')
        pipeline \
            .add_step('inner_pipln', test.pipelines.DiscardPipeline(1, name='inner_pipln') \
                      .add_step('counter', test.steps.LabelCounterStep(name='counter')))

        X = np.array([[1, 2],
                      [-1, 2],
                      [2, -1],
                      [-1, -1],
                      [2, 2],
                      [2, 2]])
        y = np.array([0, 0, 0, 1, 1, 1])
        w = np.arange(6)

        pipeline.fit(X, y, w)
        counter = pipeline.get_step('inner_pipln').get_step('counter')

        self.assertTrue(0 not in counter.n_fit)
        self.assertTrue(1 in counter.n_fit)
        self.assertEqual(counter.n_fit[1], 2)

        self.assertTrue(0 not in counter.n_transform)
        self.assertTrue(1 in counter.n_transform)
        self.assertEqual(counter.n_transform[1], 2)

        pipeline.transform(X, y, w)
        self.assertTrue(0 in counter.n_transform)
        self.assertTrue(1 in counter.n_transform)
        self.assertEqual(counter.n_transform[0], 3)
        self.assertEqual(counter.n_transform[1], 3)

    def test_set_params(self):
        pipeline = Pipeline()
        indices = [2, 4]
        pipeline \
            .add_step('scaler1', test.steps.ScaleStep(), indices) \
            .add_step('scaler2', test.steps.ScaleStep(), indices)

        pipeline.set_step_params('scaler1', {'factor': 5, })

        X = np.random.rand(50).reshape((10, 5))
        y = np.arange(10)
        w = np.arange(10)

        fit(pipeline, X, y, w)
        Xt, _, _ = transform(pipeline, X, y, w)

        X[:, indices] *= 5
        self.assertTrue(mat_eq(X, Xt))

    def test_skipping(self):
        pipeline = Pipeline()
        pipeline \
            .add_step('scaler1', test.steps.ScaleStep()) \
            .add_step('scaler2', test.steps.ScaleStep()) \
            .add_step('scaler3_4', Pipeline()
                      .add_step('scaler3', test.steps.ScaleStep())
                      .add_step('scaler4', test.steps.ScaleStep())) \
            .add_step('scaler5', test.steps.ScaleStep()) \
            .add_step('scaler6', test.steps.ScaleStep())

        pipeline.set_step_params('scaler1', {'factor': 2, })
        pipeline.set_step_params('scaler2', {'factor': 3, })
        pipeline.get_step('scaler3_4').set_step_params('scaler3', {'factor': 5, })
        pipeline.get_step('scaler3_4').set_step_params('scaler4', {'factor': 7, })
        pipeline.set_step_params('scaler5', {'factor': 11, })
        pipeline.set_step_params('scaler6', {'factor': 13, })

        X = np.random.rand(50).reshape((10, 5))
        y = np.arange(10)
        w = np.arange(10)

        fit(pipeline, X, y, w)

        Xt, _, _ = transform(pipeline, X, y, w)
        self.assertTrue(mat_eq(X * 2 * 3 * 5 * 7 * 11 * 13, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, first_step='scaler2')
        self.assertTrue(mat_eq(X * 3 * 5 * 7 * 11 * 13, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, first_step='scaler3_4')
        self.assertTrue(mat_eq(X * 5 * 7 * 11 * 13, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, first_step=['scaler3_4', 'scaler4'])
        self.assertTrue(mat_eq(X * 7 * 11 * 13, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, last_step='scaler5')
        self.assertTrue(mat_eq(X * 2 * 3 * 5 * 7 * 11, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, last_step='scaler3_4')
        self.assertTrue(mat_eq(X * 2 * 3 * 5 * 7, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, last_step=['scaler3_4', 'scaler3'])
        self.assertTrue(mat_eq(X * 2 * 3 * 5, Xt))

        Xt, _, _ = transform(pipeline, X, y, w, first_step='scaler2', last_step=['scaler3_4', 'scaler3'])
        self.assertTrue(mat_eq(X * 3 * 5, Xt))


class TestBalancedPipeline(unittest.TestCase):
    def test_stacked_step(self):
        pipeline = BalancedPipeline('wpipln')
        pipeline \
            .add_step('counter1', test.steps.LabelCounterStep(), [1, 2]) \
            .add_step('counter2', test.steps.LabelCounterStep(), [2, 3])

        X = np.random.rand(50).reshape((10, 5))
        y = np.array([0, 2, 2, 0, 1, 0, 1, 0, 2, 0])
        w = np.arange(10)

        fit(pipeline, X, y, w)
        transform(pipeline, X, y, w)
        self.assertTrue(pipeline.has_step('counter1'))
        self.assertTrue(pipeline.has_step('counter2'))

        n_fit1 = pipeline.get_step('counter1').n_fit
        n_fit2 = pipeline.get_step('counter2').n_fit
        n_trn1 = pipeline.get_step('counter1').n_transform
        n_trn2 = pipeline.get_step('counter2').n_transform

        self.assertEqual(n_fit1[0], 2)
        self.assertEqual(n_fit1[1], 2)
        self.assertEqual(n_fit1[2], 2)

        self.assertEqual(n_fit2[0], 2)
        self.assertEqual(n_fit2[1], 2)
        self.assertEqual(n_fit2[2], 2)

        self.assertEqual(n_trn1[0], 5)
        self.assertEqual(n_trn1[1], 2)
        self.assertEqual(n_trn1[2], 3)

        self.assertEqual(n_trn2[0], 5)
        self.assertEqual(n_trn2[1], 2)
        self.assertEqual(n_trn2[2], 3)

    def test_balanced_dropout(self):
        pipeline = BalancedPipeline('wpipln')
        pipeline.add_step('counter', test.steps.LabelCounterStep())
        pipeline.set_param('n_max', 2)

        X = np.random.rand(50).reshape((10, 5))
        y = np.array([0, 2, 2, 0, 1, 0, 1, 0, 2, 0])
        w = np.arange(10)

        fit(pipeline, X, y, w)
        transform(pipeline, X, y, w)

        n_fit = pipeline.get_step('counter').n_fit
        n_trn = pipeline.get_step('counter').n_transform

        self.assertEqual(sum(n_fit.values()), 6)

        self.assertEqual(n_trn[0], 5)
        self.assertEqual(n_trn[1], 2)
        self.assertEqual(n_trn[2], 3)


class TestPCAPipeline(unittest.TestCase):
    def test_standardize_step(self):
        pipeline = Pipeline()
        pipeline.add_step('std', Standardize())

        a = np.array([1., 2., 3., 4.])
        b = np.array([2., 4., 6., 8.])
        w = np.array([2., 3., 3., 4.])

        X = np.stack((a, b), axis=-1)
        y = np.zeros(4)

        pipeline.fit(X, y, w)

        step = pipeline.get_step('std')
        self.assertAlmostEqual(step.mean[0], 2.75)
        self.assertAlmostEqual(step.mean[1], 5.5)
        self.assertAlmostEqual(step.std[0], np.sqrt(14.25) / 3.)
        self.assertAlmostEqual(step.std[1], np.sqrt(57.) / 3.)

        Xt, _, _ = pipeline.transform(X, y, w)

        X = np.array([(np.array([1., 2., 3., 4.]) - 2.75) / (np.sqrt(14.25) / 3.),
                      (np.array([2., 4., 6., 8.]) - 5.5) / (np.sqrt(57.) / 3.)]).T

        self.assertTrue(mat_eq(X, Xt))

    def test_PCA(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(standardize=False))

        X = np.array([[1., 2., 3.], [2., 4., 6.]]).T
        w = np.array([1., 1., 1.])
        y = np.zeros(3)

        abspcc = lambda a, b: abs(np.corrcoef(a, b)[0][1])
        self.assertGreater(abspcc(X[:, 0], X[:, 1]), .99)

        fit(pipeline, X, y, w)

        pca = pipeline.get_step('pca')
        R = pca.R
        self.assertEqual(R.shape, (2, 2))
        self.assertTrue(np.allclose(R.dot(R.T), np.identity(R.shape[0])))

        Xt, _, _ = transform(pipeline, X, y, w)
        cov = np.cov(Xt.T)
        self.assertAlmostEqual(cov[0, 0], 5.)
        self.assertAlmostEqual(cov[1, 0], 0.)
        self.assertAlmostEqual(cov[0, 1], 0.)
        self.assertAlmostEqual(cov[1, 1], 0.)

    def test_PCA_ignore(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(ignore=[1, 2], standardize=False))

        X = np.array([[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.]]).T
        w = np.array([1., 1., 1., 1., 1.])
        y = np.array([0, 0, 0, 1, 2])

        fit(pipeline, X, y, w)

        pca = pipeline.get_step('pca')
        R = pca.R
        self.assertEqual(R.shape, (2, 2))
        self.assertTrue(np.allclose(R.dot(R.T), np.identity(R.shape[0])))

        Xt, _, _ = transform(pipeline, X, y, w)
        cov = np.cov(Xt[(y != 1) & (y != 2)].T)
        self.assertAlmostEqual(cov[0, 0], 5.)
        self.assertAlmostEqual(cov[1, 0], 0.)
        self.assertAlmostEqual(cov[0, 1], 0.)
        self.assertAlmostEqual(cov[1, 1], 0.)

    def test_StdPCA(self):
        pipeline = Pipeline()
        pipeline \
            .add_step('std', Standardize()) \
            .add_step('pca', PCA())

        n = 100
        g = lambda: np.random.rand(n)
        a = g() * 2
        b = a + (g() - .5) / 2.
        c = -a + (g() - .5) / 2.
        X = np.stack((a, b, c), axis=-1)
        y = np.zeros(n)
        w = np.ones(n)

        abspcc = lambda a, b: abs(np.corrcoef(a, b)[0][1])
        self.assertGreater(abspcc(X[:, 0], X[:, 1]), .9)
        self.assertGreater(abspcc(X[:, 0], X[:, 2]), .9)
        self.assertGreater(abspcc(X[:, 1], X[:, 2]), .9)

        fit(pipeline, X, y, w)
        Xt, _, _ = transform(pipeline, X, y, w)

        cov = np.cov(Xt.T)
        row_sums = cov.sum(axis=0)
        ncov = cov / row_sums[:, np.newaxis]

        self.assertTrue(np.allclose(ncov, np.identity(ncov.shape[0])))

    def test_PCA_standardize(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(standardize=True))

        X = np.array([[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.]]).T
        w = np.array([1., 1., 1., 1., 1.])
        y = np.array([0, 0, 0, 1, 2])

        fit(pipeline, X, y, w)
        Xt, _, _ = transform(pipeline, X, y, w)
        for mean, std in zip(Xt.mean(axis=0), Xt.std(axis=0, ddof=1)):
            self.assertAlmostEqual(mean, 0.)
            self.assertAlmostEqual(std, 1.)

    def test_PCA_weighted_standardize(self):
        pipeline = Pipeline()
        pipeline.add_step('pca', PCA(standardize=False))

        X1 = np.array([[1., 2., 3., 4., 5.], [2., 4., 6., 8., 10.]]).T
        w1 = np.array([0., 1., 2., 1., 1.])
        y1 = np.array([0, 0, 0, 1, 2])

        X2 = np.array([[2., 3., 3., 4., 5.], [4., 6., 6., 8., 10.]]).T
        w2 = np.array([1., 1., 1., 1., 1.])
        y2 = np.array([0, 0, 0, 1, 2])

        fit(pipeline, X1, y1, w1)
        Xt1, _, _ = transform(pipeline, X1, y1, w1)
        means1 = Xt1.mean(axis=0)
        stds1 = Xt1.std(axis=0, ddof=1)

        fit(pipeline, X2, y2, w2)
        Xt2, _, _ = transform(pipeline, X2, y2, w2)
        means2 = Xt1.mean(axis=0)
        stds2 = Xt1.std(axis=0, ddof=1)

        for mean1, mean2 in zip(means1, means2):
            self.assertAlmostEqual(mean1, mean2)

        for std1, std2 in zip(stds1, stds2):
            self.assertAlmostEqual(std1, std2)


class TestBinaryOverlapPCA(unittest.TestCase):
    def test_ordering(self):
        pipeline = Pipeline()
        pipeline.add_step('std', BinaryOverlapPCA())

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
        overlap1 = BinaryOverlapPCA.overlap(x1=Xt[sel1, 0], w1=w[sel1], x2=Xt[sel2, 0], w2=w[sel2],
                                            bin_edges=np.linspace(-5, 5, 101))
        overlap2 = BinaryOverlapPCA.overlap(x1=Xt[sel1, 1], w1=w[sel1], x2=Xt[sel2, 1], w2=w[sel2],
                                            bin_edges=np.linspace(-5, 5, 101))

        self.assertGreater(overlap1, .9)
        self.assertLess(overlap2, .1)


if __name__ == '__main__':
    unittest.main()

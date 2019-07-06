import unittest

import numpy as np

import test.pipelines
import test.steps
from test.helper import fit, transform, mat_eq
from wpipln.pipelines import Pipeline, BalancedPipeline
from wpipln.steps import BaseStep, Standardize, PCA


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

    def test_dropout(self):
        pipeline = BalancedPipeline('wpipln')
        pipeline.add_step('counter', test.steps.LabelCounterStep())
        pipeline.set_params({'n_max': 5})

        X = np.random.rand(50).reshape((10, 5))
        y = np.array([0, 2, 2, 0, 1, 0, 1, 0, 2, 0])
        w = np.arange(10)

        fit(pipeline, X, y, w)
        transform(pipeline, X, y, w)

        n_fit = pipeline.get_step('counter').n_fit
        n_trn = pipeline.get_step('counter').n_transform

        self.assertEqual(sum(n_fit.values()), 5)

        self.assertEqual(n_trn[0], 5)
        self.assertEqual(n_trn[1], 2)
        self.assertEqual(n_trn[2], 3)


class TestPCAPipeline(unittest.TestCase):
    def test_standardize_step(self):
        pipeline = Pipeline()
        pipeline.add_step('std', Standardize())

        a = np.arange(7).astype(float)
        b = (np.arange(7) * 2).astype(float)
        X = np.stack((a, b), axis=-1)
        y = np.arange(7)
        w = np.arange(7)

        pipeline.fit(X, y, w)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.assertAlmostEqual(mean[0], 3)
        self.assertAlmostEqual(mean[1], 6)
        self.assertAlmostEqual(std[0], 2)
        self.assertAlmostEqual(std[1], 4)

        Xt, _, _ = pipeline.transform(X, y, w)

        mean = Xt.mean(axis=0)
        std = Xt.std(axis=0)
        self.assertAlmostEqual(mean[0], 0)
        self.assertAlmostEqual(mean[1], 0)
        self.assertAlmostEqual(std[0], 1)
        self.assertAlmostEqual(std[1], 1)

    def test_StdPCA(self):
        pipeline = Pipeline()
        pipeline \
            .add_step('std', Standardize()) \
            .add_step('pca', PCA())

        g = lambda: np.random.rand(100)
        a = g() * 2
        b = a + (g() - .5) / 10.
        c = -a + (g() - .5) / 10.
        X = np.stack((a, b, c), axis=-1)
        y = np.arange(100)
        w = np.arange(100)

        abspcc = lambda a, b: abs(np.corrcoef(a, b)[0][1])
        self.assertGreater(abspcc(X[:, 0], X[:, 1]), .99)
        self.assertGreater(abspcc(X[:, 0], X[:, 2]), .99)
        self.assertGreater(abspcc(X[:, 1], X[:, 2]), .99)

        fit(pipeline, X, y, w)

        Xt, _, _ = transform(pipeline, X, y, w, last_step='std')
        self.assertTrue(all(np.abs(Xt.mean(axis=0)) < 1e-10))
        self.assertTrue(all(np.abs(Xt.std(axis=0) - 1.) < 1e-10))

        Xt, _, _ = transform(pipeline, X, y, w)
        self.assertLess(abspcc(Xt[:, 0], Xt[:, 1]), 1e-10)
        self.assertLess(abspcc(Xt[:, 0], Xt[:, 2]), 1e-10)
        self.assertLess(abspcc(Xt[:, 1], Xt[:, 2]), 1e-10)


if __name__ == '__main__':
    unittest.main()

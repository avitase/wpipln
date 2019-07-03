import unittest

import numpy as np

import test.steps
from test.helper import fit, transform, mat_eq
from wpipln.pipelines import Pipeline, BalancedPipeline
from wpipln.steps import BaseStep


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


if __name__ == '__main__':
    unittest.main()

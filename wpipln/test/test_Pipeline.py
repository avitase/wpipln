import unittest

import numpy as np

from wpipln.pipelines import Pipeline
from wpipln.steps import BaseStep
from .helper import Overwrite, Center, Scale, SkipPipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        m, n = 1000, 5
        self.X = np.random.rand(m, n)
        self.y = np.random.randint(low=0, high=3, size=m)
        self.w = np.random.rand(m)

    def fit(self, pipeline, **kwargs):
        pipeline.fit(self.X, self.y, self.w, **kwargs)

    def transform(self, pipeline, **kwargs):
        Xt, yt, wt = pipeline.transform(self.X, self.y, self.w, **kwargs)
        self.assertTrue(np.allclose(self.y, yt))
        self.assertTrue(np.allclose(self.w, wt))
        return Xt

    def test_unit_step(self):
        pipeline = Pipeline()
        pipeline.add_step('unit', BaseStep('unit'), [2, 4])
        self.assertTrue(pipeline.has_step('unit'))

        self.fit(pipeline)
        Xt = self.transform(pipeline)

        self.assertTrue(np.allclose(self.X, Xt))

    def test_Overwriter(self):
        pipeline = Pipeline()
        indices = [2, 4]
        pipeline.add_step('overwrite', Overwrite(value=0), indices)

        self.fit(pipeline)
        Xt = self.transform(pipeline)

        self.X[:, indices] = 0
        self.assertTrue(np.allclose(self.X, Xt))

    def test_indices_wildcard(self):
        pipeline = Pipeline()
        pipeline.add_step('overwrite', Overwrite(value=0), '*')

        self.fit(pipeline)
        Xt = self.transform(pipeline)

        self.X[:, :] = 0
        self.assertTrue(np.allclose(self.X, Xt))

    def test_fit_decoupling(self):
        pipeline = Pipeline()
        indices = [2, 4]
        pipeline.add_step('center', Center(), indices)

        X = np.random.rand(5, 10)
        X[:, indices[0]] = np.arange(5).astype(float)
        X[:, indices[1]] = np.arange(5).astype(float) * 2
        mean = np.mean(X[:, indices], axis=0)
        self.assertAlmostEqual(mean[0], 2.)
        self.assertAlmostEqual(mean[1], 4.)

        y = np.arange(5)
        w = np.arange(5)

        pipeline.fit(X, y, w)

        Xt, _, _ = pipeline.transform(np.ones_like(X), y, w)

        means = Xt.mean(axis=0)
        self.assertAlmostEqual(means[0], 1.)
        self.assertAlmostEqual(means[1], 1.)
        self.assertAlmostEqual(means[2], -1.)
        self.assertAlmostEqual(means[3], 1.)
        self.assertAlmostEqual(means[4], -3.)

    def test_compose(self):
        pipeline = Pipeline()
        pipeline \
            .add_step('overwrite1', Overwrite(value=1), [2, ]) \
            .add_step('overwrite2', Overwrite(value=2), [4, ]) \
            .add_step('scale', Scale(3), [2, 4])

        self.fit(pipeline)

        Xt = self.transform(pipeline, last_step='overwrite2')
        self.assertTrue(np.all(Xt[:, 2] == 1))
        self.assertTrue(np.all(Xt[:, 4] == 2))
        self.assertTrue(np.allclose(self.X[:, [0, 1, 3]], Xt[:, [0, 1, 3]]))

        Xt = self.transform(pipeline)
        self.assertTrue(np.all(Xt[:, 2] == 3))
        self.assertTrue(np.all(Xt[:, 4] == 6))
        self.assertTrue(np.allclose(self.X[:, [0, 1, 3]], Xt[:, [0, 1, 3]]))

    def test_inner_pipeline(self):
        pipeline = SkipPipeline(skip_rows=[2, 3], name='outer_pipln')
        pipeline.add_step('inner_pipln', SkipPipeline(skip_rows=[2, ], name='inner_pipln') \
                          .add_step('center', Center()))

        X = np.array([[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]])
        y = np.random.rand(5)
        w = np.random.rand(5)

        pipeline.fit(X, y, w)

        Xt, yt, wt = pipeline.transform(X, y, w)
        self.assertTrue(np.allclose(Xt, [[-1, -1], [1, 1], [3, 3], [5, 5], [7, 7]]))
        self.assertTrue(np.allclose(y, yt))
        self.assertTrue(np.allclose(w, wt))

    def test_set_params(self):
        pipeline = Pipeline()
        pipeline \
            .add_step('scaler1', Scale()) \
            .add_step('scaler2', Scale())

        pipeline.set_step_params('scaler1', {'factor': 2, })
        pipeline.set_step_params('scaler2', {'factor': 3, })

        self.fit(pipeline)
        Xt = self.transform(pipeline)

        self.assertTrue(np.allclose(self.X * 6, Xt))

    def test_skipping(self):
        pipeline = Pipeline()
        pipeline \
            .add_step('scaler1', Scale(factor=2)) \
            .add_step('scaler2', Scale(factor=3)) \
            .add_step('scaler3_4', Pipeline()
                      .add_step('scaler3', Scale(5))
                      .add_step('scaler4', Scale(7))) \
            .add_step('scaler5', Scale(11)) \
            .add_step('scaler6', Scale(13))

        self.fit(pipeline)

        Xt = self.transform(pipeline)
        self.assertTrue(np.allclose(self.X * 2 * 3 * 5 * 7 * 11 * 13, Xt))

        Xt = self.transform(pipeline, first_step='scaler2')
        self.assertTrue(np.allclose(self.X * 3 * 5 * 7 * 11 * 13, Xt))

        Xt = self.transform(pipeline, first_step='scaler3_4')
        self.assertTrue(np.allclose(self.X * 5 * 7 * 11 * 13, Xt))

        Xt = self.transform(pipeline, first_step=['scaler3_4', 'scaler4'])
        self.assertTrue(np.allclose(self.X * 7 * 11 * 13, Xt))

        Xt = self.transform(pipeline, last_step='scaler5')
        self.assertTrue(np.allclose(self.X * 2 * 3 * 5 * 7 * 11, Xt))

        Xt = self.transform(pipeline, last_step='scaler3_4')
        self.assertTrue(np.allclose(self.X * 2 * 3 * 5 * 7, Xt))

        Xt = self.transform(pipeline, last_step=['scaler3_4', 'scaler3'])
        self.assertTrue(np.allclose(self.X * 2 * 3 * 5, Xt))

        Xt = self.transform(pipeline, first_step='scaler2', last_step=['scaler3_4', 'scaler3'])
        self.assertTrue(np.allclose(self.X * 3 * 5, Xt))

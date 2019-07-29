import unittest

import numpy as np

from wpipln.pipelines import Pipeline
from wpipln.steps import Standardizer


class TestStandardizer(unittest.TestCase):
    def test_standardize(self):
        pipeline = Pipeline()
        pipeline.add_step('std', Standardizer())

        X = np.array([[1., 2.],
                      [2., 4.],
                      [3., 6.],
                      [4., 8.]])
        y = np.zeros(4)
        w = np.array([2., 3., 3., 4.])

        pipeline.fit(X, y, w)

        step = pipeline.get_step('std')
        self.assertAlmostEqual(step.mean[0], 2.75)
        self.assertAlmostEqual(step.mean[1], 5.5)
        self.assertAlmostEqual(step.std[0], np.sqrt(14.25) / 3.)
        self.assertAlmostEqual(step.std[1], np.sqrt(57.) / 3.)

        Xt, _, _ = pipeline.transform(X, y, w)

        X = np.array([(np.array([1., 2., 3., 4.]) - 2.75) / (np.sqrt(14.25) / 3.),
                      (np.array([2., 4., 6., 8.]) - 5.5) / (np.sqrt(57.) / 3.)]).T

        self.assertTrue(np.allclose(X, Xt))

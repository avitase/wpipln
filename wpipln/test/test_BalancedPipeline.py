import unittest

import numpy as np

from wpipln.pipelines import BalancedPipeline
from .helper import Center


class TestBalancedPipeline(unittest.TestCase):
    def test_default(self):
        pipeline = BalancedPipeline('wpipln')
        pipeline.add_step('center', Center())

        X = np.random.rand(10, 5)
        y = np.array([0, 2, 2, 0, 1, 0, 1, 0, 2, 0])
        w = np.ones(10)

        pipeline.fit(X, y, w)
        Xt, yt, wt = pipeline.transform(X, y, w)
        self.assertTrue(np.allclose(y, yt))
        self.assertTrue(np.allclose(w, wt))

        mean = np.sum(X[[0, 1, 2, 3, 4, 6], :], axis=0) / 6.
        self.assertTrue(np.allclose(X - mean[np.newaxis, :], Xt))

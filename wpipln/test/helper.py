import numpy as np

from wpipln.pipelines import Pipeline
from wpipln.steps import BaseStep


class Overwrite(BaseStep):
    def __init__(self, value, name='Overwrite'):
        super(Overwrite, self).__init__(name)
        self.value = value

    def transform(self, X, y, w):
        assert self.is_fitted
        return np.ones_like(X) * self.value, y, w


class Center(BaseStep):
    def __init__(self, name='Center'):
        super(Center, self).__init__(name)
        self.mean = None

    def fit(self, X, y, w):
        self.mean = X.mean(axis=0)
        self.is_fitted = True

    def transform(self, X, y, w):
        assert self.is_fitted
        return X - self.mean[np.newaxis, :], y, w


class Scale(BaseStep):
    def __init__(self, factor=None, name='Scale'):
        super(Scale, self).__init__(name)
        self.params['factor'] = factor

    def transform(self, X, y, w):
        assert self.is_fitted
        return X * self.params['factor'], y, w


class SkipPipeline(Pipeline):
    def __init__(self, skip_rows, name='SkipPipeline'):
        super(SkipPipeline, self).__init__(name)
        self.skip_rows = skip_rows

    def filter(self, X, y, w):
        sel = super(SkipPipeline, self).filter(X, y, w)

        accept_row = np.array([True, ] * len(y))
        accept_row[self.skip_rows] = False

        return sel & accept_row

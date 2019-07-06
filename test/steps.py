import numpy as np

from wpipln.steps import BaseStep


class ZeroXStep(BaseStep):
    def __init__(self, name='ZeroXStep'):
        super(ZeroXStep, self).__init__(name)

    def transform(self, X, y, w):
        return np.zeros(X.shape), y, w


class CenterStep(BaseStep):
    def __init__(self, name='CenterStep'):
        super(CenterStep, self).__init__(name)

    def fit(self, X, y, w):
        self.mean = np.mean(X, axis=0)

    def transform(self, X, y, w):
        _, m = X.shape
        assert len(self.mean) == m

        for i in range(m):
            X[:, i] -= self.mean[i]

        return X, y, w


class StdScaleStep(BaseStep):
    def __init__(self, name='StdScaleStep'):
        super(StdScaleStep, self).__init__(name)

    def fit(self, X, y, w):
        self.std = np.std(X, axis=0)

    def transform(self, X, y, w):
        _, m = X.shape
        assert len(self.std) == m

        for i in range(m):
            X[:, i] /= self.std[i]

        return X, y, w


class ScaleStep(BaseStep):
    def __init__(self, name='ScaleStep'):
        super(ScaleStep, self).__init__(name)

    def transform(self, X, y, w):
        scale_factor = self.params['factor'] if 'factor' in self.params else 1.

        return X * scale_factor, y, w


class LabelCounterStep(BaseStep):
    def __init__(self, name='LabelCounterStep'):
        super(LabelCounterStep, self).__init__(name)
        self.n_fit = dict()
        self.n_transform = dict()

    def count(y):
        labels = np.unique(y)
        return {label: np.sum(y == label) for label in labels}

    def fit(self, X, y, w):
        assert X.shape[0] == len(y) == len(w), f'{X.shape}, {len(y)}, {len(w)}'

        self.n_fit = LabelCounterStep.count(y)

    def transform(self, X, y, w):
        assert X.shape[0] == len(y) == len(w)

        self.n_transform = LabelCounterStep.count(y)

        return X, y, w

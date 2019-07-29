import numpy as np

from .BaseStep import BaseStep
from .helper import average, standard_deviation


class Standardizer(BaseStep):
    def __init__(self, name='Standardizer'):
        super(Standardizer, self).__init__(name)
        self.mean = None
        self.std = None

    def fit(self, X, y, w):
        self.mean = average(X, w)
        self.std = standard_deviation(X, w)

    def transform(self, X, y, w):
        assert self.mean is not None
        assert self.std is not None

        return (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :], y, w

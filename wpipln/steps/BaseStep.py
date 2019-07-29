import numpy as np


class BaseStep:
    def __init__(self, name):
        self.name = name
        self.params = dict()

    def __str__(self):
        return f'Pipeline step \'{self.name}\''

    def set_param(self, key, param):
        self.params[key] = param

    def set_params(self, params):
        for key in params:
            self.set_param(key, params[key])

    def filter(self, X, y, w):
        accept_all = np.array([True, ] * len(y))
        return accept_all

    def fit(self, X, y, w):
        pass

    def transform(self, X, y, w):
        return X, y, w

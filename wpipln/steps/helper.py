import numpy as np


def average(X, w):
    return np.average(X, axis=0, weights=w)


def variance(X, w):
    n, _ = X.shape
    avg = average(X, w)
    return np.average((X - avg[np.newaxis, :]) ** 2, axis=0, weights=w) * n / (n - 1)


def standard_deviation(X, w):
    var = variance(X, w)
    return np.sqrt(var)

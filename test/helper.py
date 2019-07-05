import numpy as np


def fit(pipeline, X, y, w):
    pipeline.fit(np.array(X, copy=True), np.array(y, copy=True), np.array(w, copy=True))


def transform(pipeline, X, y, w, first_step=None, last_step=None):
    return pipeline.transform(np.array(X, copy=True), np.array(y, copy=True), np.array(w, copy=True),
                              first_step=first_step, last_step=last_step)


def mat_eq(M1, M2):
    if M1.shape != M2.shape: return False
    return np.allclose(M1, M2)

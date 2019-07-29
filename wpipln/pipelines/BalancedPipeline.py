import numpy as np

from .Pipeline import Pipeline


class BalancedPipeline(Pipeline):
    def __init__(self, name='BalancedPipeline'):
        super(BalancedPipeline, self).__init__(name)

    def filter(self, X, y, w):
        labels = np.unique(y)
        n = min(np.sum(y == label) for label in labels)

        return Pipeline.balanced_truncate(X, y, w, n)

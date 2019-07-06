from wpipln.pipelines import Pipeline


class DiscardPipeline(Pipeline):
    def __init__(self, x, name='DiscardPipeline'):
        super(DiscardPipeline, self).__init__(name)
        self.x = x

    def filter(self, X, y, w):
        sel = super(DiscardPipeline, self).filter(X, y, w)

        n, m = X.shape
        for i in range(m):
            sel &= X[:, i] != self.x

        return sel

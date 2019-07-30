import numpy as np

from wpipln.pipelines import Pipeline
from wpipln.steps import BaseStep


class TestStep(BaseStep):
    def __init__(self):
        super(TestStep, self).__init__('TestStep')
        self.mean = None

    def fit(self, X, y, w):
        self.mean = X.mean()

    def transform(self, X, y, w):
        X = np.ones_like(X) * self.mean
        return X, y, w


def generate_data(m=10, n=5):
    X = np.random.rand(m, n)
    y = np.random.randint(low=0, high=2, size=m)
    w = np.random.rand(m)
    return X, y, w


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline \
        .add_step('step1', TestStep(), indices=[1, 3]) \
        .add_step('step2', TestStep(), indices=[2, 4])

    X1, y1, w1 = generate_data()
    pipeline.fit(X1, y1, w1)

    X2, y2, w2 = generate_data()
    Xt, yt, wt = pipeline.transform(X2, y2, w2)

    print(Xt)

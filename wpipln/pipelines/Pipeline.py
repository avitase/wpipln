import numpy as np


class Pipeline:
    def __init__(self, name='Pipeline', steps=[]):
        self.name = name
        self.params = dict()
        self.is_fitted = False

        self.steps = []
        names = []
        for name, step, indices in steps:
            if name not in names:
                names.append(name)
                self.steps.append((name, step, indices))
            else:
                raise ValueError('names have to be unique')

    def add_step(self, name, step, indices='*'):
        self.steps.append((name, step, indices))
        return self

    def has_step(self, name):
        return name in {name for (name, step, indices) in self.steps}

    def get_step(self, name):
        for step_name, step, _ in self.steps:
            if step_name == name:
                return step

        return None

    def __str__(self):
        str_repr = f'{self.name}, steps: {{'
        str_repr += ', '.join(f'(({step}), indices={indices})' for name, step, indices in self.steps)
        str_repr += '}'
        return str_repr

    def set_param(self, key, param):
        self.params[key] = param

    def set_params(self, params):
        for key in params:
            self.set_param(key, params[key])

    def set_step_param(self, step_name, key, param):
        assert step_name in [name for (name, _, _) in self.steps]

        for name, step, _ in self.steps:
            if name == step_name:
                step.set_param(key, param)

    def set_step_params(self, name, params):
        for key in params:
            self.set_step_param(name, key, params[key])

    @staticmethod
    def balanced_truncate(X, y, w, n_max):
        n = len(y)
        sel = np.array([False, ] * n)

        labels = np.unique(y)
        for label in labels:
            idx = (y == label)
            v = np.zeros(n)
            v[idx] = np.arange(np.sum(idx)) + 1
            sel |= (v > 0) & (v <= n_max)

        return sel

    def filter(self, X, y, w):
        n, _ = X.shape

        n_max = self.params['n_max'] if 'n_max' in self.params else n
        if n_max == -1 or n_max is None or n_max >= n:
            return np.array([True, ] * n)

        assert n_max > 0, f'n_max is {n_max} but should be positive'

        return Pipeline.balanced_truncate(X, y, w, n_max)

    def fit(self, X, y, w, copy=True):
        assert X.shape[0] == len(y) == len(w), f'{X.shape}, {len(y)}, {len(w)}'

        X_cpy = np.array(X, copy=copy)
        y_cpy = np.array(y, copy=copy)
        w_cpy = np.array(w, copy=copy)

        sel = self.filter(X_cpy, y_cpy, w_cpy)
        X_cpy, y_cpy, w_cpy = X_cpy[sel, :], y_cpy[sel], w_cpy[sel]

        for _, step, indices in self.steps:
            ids = np.arange(X.shape[1]) if indices == '*' else indices

            step.fit(X_cpy[:, ids], y_cpy, w_cpy)

            sel = step.filter(X_cpy[:, ids], y_cpy, w_cpy)
            X_cpy, y_cpy, w_cpy = X_cpy[sel, :], y_cpy[sel], w_cpy[sel]
            X_trns, y_trns, w_trns = step.transform(X_cpy[:, ids], y_cpy, w_cpy)

            assert X_trns.shape[1] == len(ids), f'{X_trns.shape}, {len(ids)}'
            assert X_trns.shape[0] == len(y_trns) == len(w_trns), f'{X_trns.shape}, {len(y_trns)}, {len(w_trns)}'
            assert X_trns.shape[0] == X_cpy.shape[0], f'{X_trns.shape[0]}, {X.shape[0]}'

            X_cpy[:, ids] = X_trns
            y_cpy = y_trns
            w_cpy = w_trns

        self.is_fitted = True

    def transform(self, X, y=None, w=None, copy=True, first_step=None, last_step=None):
        assert self.is_fitted, 'transform can only be called after wpipln was fitted'

        if y is not None:
            assert X.shape[0] == len(y), f'{X.shape}, {len(y)}'

        if w is not None:
            assert X.shape[0] == len(w), f'{X.shape}, {len(w)}'

        X_cpy = np.array(X, copy=copy)
        y_cpy = np.array(y, copy=copy) if y is not None else None
        w_cpy = np.array(w, copy=copy) if w is not None else None

        split = lambda x: (x, []) if isinstance(x, str) else (x[0], x[1:])
        step_number = {name: n for n, (name, _, _) in enumerate(self.steps)}

        if first_step is None:
            first, first_tail = 0, None
        else:
            head, tail = split(first_step)
            first = step_number[head]
            first_tail = tail if len(tail) > 0 else None

        if last_step is None:
            last, last_tail = len(self.steps) - 1, None
        else:
            head, tail = split(last_step)
            last = step_number[head]
            last_tail = tail if len(tail) > 0 else None

        assert first <= last

        first_name, _, _ = self.steps[first]
        last_name, _, _ = self.steps[last]
        for name, step, indices in self.steps[first:last + 1]:
            is_first = (name == first_name)
            is_last = (name == last_name)

            ids = np.arange(X.shape[1]) if indices == '*' else indices

            if is_first and first_tail:
                trns = step.transform(X_cpy[:, ids], y_cpy, w_cpy, first_step=first_tail)
            elif is_last and last_tail:
                trns = step.transform(X_cpy[:, ids], y_cpy, w_cpy, last_step=last_tail)
            else:
                trns = step.transform(X_cpy[:, ids], y_cpy, w_cpy)

            X_trns, y_trns, w_trns = trns
            assert X_trns.shape[1] == len(ids), f'{X_trns.shape}, {len(ids)}'
            assert X_trns.shape[0] == X_cpy.shape[0], f'{X_trns.shape[0]}, {X.shape[0]}'

            if y_trns is not None:
                assert X_trns.shape[0] == len(y_trns), f'{X_trns.shape}, {len(y_trns)}'

            if w_trns is not None:
                assert X_trns.shape[0] == len(w_trns), f'{X_trns.shape}, {len(w_trns)}'

            X_cpy[:, ids] = X_trns
            y_cpy = y_trns
            w_cpy = w_trns

        return X_cpy, y_cpy, w_cpy

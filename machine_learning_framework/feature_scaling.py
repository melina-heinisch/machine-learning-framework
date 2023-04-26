import numpy as np


class StandardScaler:

    def fit(self, values):
        self.mean = np.mean(values, axis=0)
        self.std = np.std(values, axis=0)

        # the standard deviation can be 0 in certain cases,
        #  which provokes 'devision-by-zero' errors; we can
        #  avoid this by adding a small amount if std==0
        self.std[self.std == 0] = 0.00001

    def transform(self, values):
        values = values - self.mean
        values = values / self.std
        return values

    def inverse_transform(self, values):
        values = values * self.std + self.mean
        return values


class NormalScaler:
    def fit(self, values):
        self.min = np.min(values, axis=0)
        self.max = np.max(values, axis=0) - self.min

        # the max can be 0 in certain cases,
        # which provokes 'devision-by-zero' errors; we can
        # avoid this by adding a small amount if max==0
        self.max = self.max.astype(float)
        self.max[self.max == 0] = 0.00001

    def transform(self, values):
        values = (values - self.min) / self.max
        return values

    def inverse_transform(self, values):
        values = values * self.max + self.min
        return values

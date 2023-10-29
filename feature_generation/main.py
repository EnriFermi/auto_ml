from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance
import numpy as np

class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x_set, y_set=None):
        self.cov_mat = empirical_covariance(x_set)
        print(self.cov_mat)
        return self

    def transform(self, x_set):
        return x_set


if __name__ == '__main__':
    x_set = np.array([
        [1, 1, 1],
        [0, 1, 0],
        [2, 1, 2],
        [7, 1, 7]
    ])
    feat_gen = FeatureGenerationTransformer()
    feat_gen.fit(x_set)
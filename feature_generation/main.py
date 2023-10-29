from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance
import numpy as np

EPS = 1e-20

class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x_set, y_set=None):
        self.cov_mat = empirical_covariance(x_set)
        self.std_arr = np.std(x_set, axis=0).reshape((1,-1))
        self.std_mat = self.std_arr.T @ self.std_arr
        self.corr_mat = self.cov_mat / (self.std_mat + EPS)
        
        return self

    def transform(self, x_set):
        return x_set


if __name__ == '__main__':
    x_set = np.array([
        [1, 1, -1],
        [0, 13, 0],
        [2, 15, -2],
        [7, 16, -7]
    ])
    feat_gen = FeatureGenerationTransformer()
    feat_gen.fit(x_set)
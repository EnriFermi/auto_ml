from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance
from sklearn. preprocessing import normalize
import numpy as np

EPS = 1e-20
THRESHOLD = 0.3

class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def correlation_create(self, x_set):
        cov_mat = empirical_covariance(x_set)
        std_arr = np.std(x_set, axis=0).reshape((1,-1))
        std_mat = std_arr.T @ std_arr
        return cov_mat / (std_mat + EPS)

    def fit(self, x_set, y_set=None):
        self.corr_mat = self.correlation_create(x_set)
        return self

    def standard_generation(self, x_set, count_features):
        x_set = np.append(x_set, np.exp(x_set[:,:count_features]), axis=1)
        x_set = np.append(x_set, np.log(x_set[:,:count_features] + 1), axis=1)

        self.corr_mat = self.correlation_create(x_set)

        return x_set

    def correlation_generation(self, x_set):
        for feat_1_ind in range(self.corr_mat.shape[0]):
            for feat_2_ind in range(feat_1_ind, self.corr_mat[feat_1_ind].shape[0]):
                if abs(self.corr_mat[feat_1_ind][feat_2_ind]) < THRESHOLD:
                    x_set = np.append(x_set, (x_set[:,feat_1_ind] + x_set[:,feat_2_ind]).reshape(-1,1), axis=1)
                    x_set = np.append(x_set, (x_set[:,feat_1_ind] - x_set[:,feat_2_ind]).reshape(-1,1), axis=1)
                    x_set = np.append(x_set, (x_set[:,feat_1_ind] * x_set[:,feat_2_ind]).reshape(-1,1), axis=1)
        
        self.corr_mat = self.correlation_create(x_set)

        return x_set

    def transform(self, x_set):
        x_set = self.standard_generation(x_set, x_set.shape[1])
        x_set = self.correlation_generation(x_set)

        return x_set


if __name__ == '__main__':
    x_set = np.array([
        [1, 1, -1],
        [0, 13, 0],
        [2, 15, -2],
        [7, 16, -7],
        [7, 17, -7],
        [7, 18, -7],
        [10, 19, -10],
        [7, 20, -7],
        [90.2, 21, -90.2],
        [90.2, 22, -90.2],
        [142.1, 23, -142.1],
        [7, 24, -7],
        [7, 25, -7],
        [7, 26, -7],
        [7, 27, -7],
        [7, 28, -7],
        [7, 29, -7]
    ])
    x_set = normalize(x_set, axis= 0 , norm='l1')
    feat_gen = FeatureGenerationTransformer()
    print(feat_gen.fit_transform(x_set))
    #print(feat_gen.fit_transform(x_set))
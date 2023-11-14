from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance
from sklearn. preprocessing import normalize
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import pandas as pd


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

import numpy as np

EPS = 1e-20
EPS_LOG = 1e-3
THRESHOLD = 0.2

class FeatureGenerationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, thr=THRESHOLD, features_mask=None):
        self.thr = thr
        self._features_mask = features_mask
    
    #Construction of a matrix of correlation coefficients of features
    def _correlation_create(self, x_set):
        cov_mat = empirical_covariance(x_set)
        std_arr = np.std(x_set, axis=0).reshape((1,-1))
        std_mat = std_arr.T @ std_arr
        return cov_mat / (std_mat + EPS)

    def fit(self, x_set, y_set=None):
        if self._features_mask is None:
            self._features_mask = np.arange(x_set.shape[1])
            
        self._corr_mat = self._correlation_create(x_set)
        return self

    #Generating features using standard functions
    def _standard_generation(self, x_set, features_mask=None):
        if features_mask is None:
            features_mask = self._features_mask
        #exponent
        x_set = np.concatenate([x_set, np.exp(x_set[:,features_mask])], axis=1)
        #logarithm
        x_set = np.concatenate([x_set, np.where(x_set[:,features_mask] <= -1, 
                                                np.log(EPS_LOG), 
                                                np.log(x_set[:,features_mask] + 1, 
                                                       where=x_set[:,features_mask] > -1))], axis=1)
        #x^2
        x_set = np.concatenate([x_set, np.power(x_set[:,features_mask], 2)], axis=1)
        #x^3
        x_set = np.concatenate([x_set, np.power(x_set[:,features_mask], 3)], axis=1)
        #x^0.5 
        x_set = np.concatenate([x_set, np.where(x_set[:,features_mask] < 0, 
                                                -np.power(-x_set[:,features_mask], 0.5, 
                                                          where=x_set[:,features_mask] < 0), 
                                                np.power(x_set[:,features_mask], 0.5, 
                                                          where=x_set[:,features_mask] >= 0))], axis=1)

        return x_set

    #Generating features from two that have a correlation coefficient less than the threshold
    def _correlation_generation(self, x_set, features_mask=None):
        if features_mask is None:
            features_mask = self._features_mask

        for feat_ind in tqdm(range(self._corr_mat[:,features_mask][features_mask,:].shape[1])):
            feature_filter = np.concatenate([np.full((feat_ind,), False), 
                                             abs(self._corr_mat[:,features_mask][features_mask,:][feat_ind][feat_ind:]) < self.thr, 
                                             np.full((x_set[:,features_mask].shape[1] - self._corr_mat[:,features_mask][features_mask,:].shape[1],), False)])
            #x1 + x2
            x_set = np.concatenate([x_set, x_set[:,features_mask].T[0].reshape(-1, 1) + x_set[:,features_mask].T[feature_filter].T], axis=1)
            #x1 - x2
            x_set = np.concatenate([x_set, x_set[:,features_mask].T[0].reshape(-1, 1) - x_set[:,features_mask].T[feature_filter].T], axis=1)
            #x1 * x2
            x_set = np.concatenate([x_set, x_set[:,features_mask].T[0].reshape(-1, 1) * x_set[:,features_mask].T[feature_filter].T], axis=1)
            #x1 / x2
            x_set = np.concatenate([x_set, np.divide(np.repeat(x_set[:,features_mask].T[0].reshape(-1, 1), 
                                                               np.count_nonzero(feature_filter), axis=1), 
                                                     x_set[:,features_mask].T[feature_filter].T, 
                                                     out=np.repeat(x_set[:,features_mask].T[0].reshape(-1, 1), 
                                                                   np.count_nonzero(feature_filter), axis=1),
                                                     where=x_set[:,features_mask].T[feature_filter].T != 0)], axis=1)

        return x_set

    def transform(self, x_set):
        x_set = self._standard_generation(x_set)
        print(x_set.shape)
        x_set = self._correlation_generation(x_set)

        return x_set


if __name__ == '__main__':
    X, y = make_classification(
    n_samples=100000, n_features=50, n_informative=40, n_redundant=5,
    random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    

    result = scaler.fit_transform(X_train)
    feat_gen = FeatureGenerationTransformer(thr=0.01)
    start_time = time.time()
    result_2 = feat_gen.fit_transform(result)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Время генерации:', elapsed_time)
    print(result_2.shape)
    print(result_2)

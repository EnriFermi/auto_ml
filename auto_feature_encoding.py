from numpy import ndarray
from sklearn.datasets import load_breast_cancer
from IPython.display import display 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, categ_rate=0.2):
        self.encoders = {'one_hot': OneHotEncoder}
        self.category_rate = categ_rate
        self.categ_features_inds = None
        self.encoder = None

    def fit(self, x, y=None, **fit_params):
        self.categ_features_inds = self._get_categorical_inds(x)
        if "enc_type" in fit_params.keys():
            self.encoder = self.encoders[fit_params["enc_type"]]()
        
        del fit_params["enc_type"]

        if self.encoder is not None:
            self.encoder = self.encoder.fit(x[:, self.categ_features_inds], y, **fit_params)
        return self
    
    def transform(self, x):
        if self.encoder is None:
            return x

        encoded = self.encoder.transform(x[:, self.categ_features_inds]).toarray()
        x = np.delete(x, self.categ_features_inds, 1)
        x = np.concatenate((x, encoded), axis=1)
        return x

    def _get_categorical_inds(self, data):
        categorical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] < data.shape[0] * self.category_rate:
                categorical_features.append(num)
        return categorical_features     



class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_rate=0.2):
        self.normalizers = {"minmax": MinMaxScaler}
        self.feature_rate = feature_rate
        self.num_features_inds = None
        self.normalizer = None

    
    def fit(self, x, y=None, **fit_params):
        self.num_features_inds = self._get_numerical_inds(x)

        if "normalize_type" in fit_params.keys():
            self.normalizer = self.normalizers[fit_params["normalize_type"]]()
        
        del fit_params["normalize_type"]

        if self.normalizer is not None:
            self.normalizer = self.normalizer.fit(x[:, self.num_features_inds], y, **fit_params)
        return self

    def transform(self, x):
        if self.normalizer is None:
            return x

        normalized = self.normalizer.transform(x[:, self.num_features_inds])
        x[:, self.num_features_inds] = normalized
        return x

    def _get_numerical_inds(self, data):
        numerical_features_inds = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] >= data.shape[0] * self.feature_rate:
                numerical_features_inds.append(num)
        return numerical_features_inds   


def main():
    data = load_breast_cancer()
    x = data.data
    y = data.target

    for i in range(x.shape[0]):
        x[i] = np.array(list(map(int, x[i]))) # Example dataset, turn real features into categories

    sc = Normalizer()
    enc = Encoder()
    result = sc.fit_transform(x, normalize_type="minmax")
    result2 = enc.fit_transform(result, enc_type="one_hot")
    print(pd.DataFrame(result2, dtype='float32').head())


if __name__ == '__main__':
    main()
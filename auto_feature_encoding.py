from numpy import ndarray
from sklearn.datasets import load_breast_cancer
from IPython.display import display 
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, categ_rate=0.2):
        self.encoders = {'one_hot': OneHotEncoder}
        self.category_rate = categ_rate
        self.categ_features_inds = None
        self.encoder = None

    def fit(self, X, y=None, **fit_params):
        self.categ_features_inds = self._get_categorical_inds(X)
        if "enc_type" in fit_params.keys():
            self.encoder = self.encoders[fit_params["enc_type"]]()
        
        del fit_params["enc_type"]

        if self.encoder is not None:
            self.encoder = self.encoder.fit(X[:, self.categ_features_inds], y, **fit_params)
        return self
    
    def transform(self, X):
        if self.encoder is None:
            return X

        encoded = self.encoder.transform(X[:, self.categ_features_inds]).toarray()
        X = np.delete(X, self.categ_features_inds, 1)
        X = np.concatenate((X, encoded), axis=1)
        return X

    def _get_categorical_inds(self, data):
        categorical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] < data.shape[0] * self.category_rate:
                categorical_features.append(num)
        return categorical_features     


def main():
    data = load_breast_cancer()
    x = data.data
    y = data.target

    for i in range(x.shape[0]):
        x[i] = np.array(list(map(int, x[i]))) # Example dataset, turn real features into categories

    enc = Encoder()
    result = enc.fit_transform(x, enc_type="one_hot")
    print(pd.DataFrame(result, dtype='float32').head())


if __name__ == '__main__':
    main()
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, categ_rate=0.2):
        self._encoders = {'one_hot': OneHotEncoder, 'label': LabelEncoder}
        self.category_rate = categ_rate
        self.categ_features_inds = None
        self.encoder = None
        self._fit_args_count = 2
        self._enc_fit_arg1 = ['label']

    def fit(self, x, y=None, **fit_params):
        if "enc_type" in fit_params.keys():
            self.encoder = self._encoders[fit_params["enc_type"]]()
        
        if fit_params["enc_type"] in self._enc_fit_arg1:
            self._fit_args_count = 1
        
        del fit_params["enc_type"]

        if self.encoder is not None:
            match self._fit_args_count:
                case 1: 
                    self.encoder = self.encoder.fit(x)
                case 2:
                    self.categ_features_inds = self._get_categorical_inds(x)
                    self.encoder = self.encoder.fit(x[:, self.categ_features_inds], y, **fit_params)

        return self
    
    def transform(self, x):
        if self.encoder is None:
            return x
        match self._fit_args_count:
            case 1:
                x = self.encoder.transform(x)
            case 2:
                encoded = self.encoder.transform(x[:, self.categ_features_inds]).toarray()
                x = np.delete(x, self.categ_features_inds, 1)
                x = np.concatenate((x, encoded), axis=1)
        return x
    
    def get_enc_types(self):
        return list(self._encoders.keys())

    def _get_categorical_inds(self, data):
        categorical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] < data.shape[0] * self.category_rate:
                categorical_features.append(num)
        return categorical_features     



class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_rate=0.2):
        self._normalizers = {"minmax": MinMaxScaler}
        self.feature_rate = feature_rate
        self.num_features_inds = None
        self.normalizer = None

    
    def fit(self, x, y=None, **fit_params):
        self.num_features_inds = self._get_numerical_inds(x)

        if "normalize_type" in fit_params.keys():
            self.normalizer = self._normalizers[fit_params["normalize_type"]]()
        
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
    
    def get_normalaizer_types(self):
        return list(self._normalizers.keys())

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

    x = sc.fit_transform(x, normalize_type="minmax")
    x = enc.fit_transform(x, enc_type="one_hot")
    y = enc.fit_transform(y, enc_type="label")

    print('Possible encoders: ', enc.get_enc_types())

    x_df = pd.DataFrame(x, dtype='float32')
    y_df = pd.DataFrame(y,  dtype='int')
    df = pd.concat((x_df, y_df), axis=1)
    print(df)


if __name__ == '__main__':
    main()
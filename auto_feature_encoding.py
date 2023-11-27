from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import numpy as np
import pandas as pd


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder='one_hot', category_rate=0.2, drop=True, **encoder_params):
        self._encoders = {'one_hot': ce.OneHotEncoder, 
                          'target': ce.TargetEncoder, 
                          'count': ce.CountEncoder,
                          'hashing': ce.HashingEncoder,
                          'binary': ce.BinaryEncoder,
                          'ordinal': ce.OrdinalEncoder,
                          'gray': ce.GrayEncoder}
        self._encoder_name = encoder
        self._encoder = self._encoders[encoder](**encoder_params)
        self._drop = drop
        self._categorical_columns_ind = None
        self.category_rate = category_rate

    def fit(self, x, y=None, **fit_params):
        self._categorical_columns_ind = self.get_categorical_columns_inds(x)
        if self._categorical_columns_ind.shape[0] == 0:
            return self        
        x_cat = x[:, self._categorical_columns_ind]

        self._encoder.fit(pd.DataFrame(x_cat, dtype='category'), y, **fit_params)
        return self

    def transform(self, x):
        if self._categorical_columns_ind is None or not self._categorical_columns_ind.shape[0]:
            return x
        
        x_cat_transformed = x[:, self._categorical_columns_ind]
        x_cat_transformed = self._encoder.transform(pd.DataFrame(x_cat_transformed, dtype='category'))
        x_cat_transformed = x_cat_transformed.to_numpy()

        x_copy = x.copy()

        if x_cat_transformed.shape[1] == self._categorical_columns_ind.shape[0]:
            x_copy[:, self._categorical_columns_ind] = x_cat_transformed
        elif self._drop:
            x_copy = np.delete(x, self._categorical_columns_ind, axis=1)
            x_copy = np.concatenate((x, x_cat_transformed), axis=1)
        else:
            x_copy = np.concatenate((x, x_cat_transformed), axis=1)
        
        return x_copy


    def get_encoder_name(self):
        return self._encoder_name

    def get_available_encoders(self):
        return np.array(list(self._encoders.keys()))

    def get_categorical_columns_inds(self, data):
        categorical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] < data.shape[0] * self.category_rate:
                categorical_features.append(num)
        return np.array(categorical_features)
    

class NumericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None, **fit_params):
        pass

    def transform(self, x):
        pass


class CustomNormalizer(BaseEstimator, TransformerMixin):
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
        else:
            self.normalizer = MinMaxScaler()

        if len(self.num_features_inds) != 0:
            self.normalizer = self.normalizer.fit(x[:, self.num_features_inds], y, **fit_params)
        else:
            self.normalizer = None
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

    x = np.array([[round(col, 1) for col in row] for row in x])

    cat_enc = CategoricalEncoder(encoder='target')
    result = cat_enc.fit_transform(x, y)


if __name__ == '__main__':
    main()

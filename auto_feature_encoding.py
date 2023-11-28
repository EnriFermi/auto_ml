from sklearn.datasets import load_breast_cancer
import sklearn.preprocessing as skpr
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import numpy as np
import pandas as pd


class CircularEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, round_precision=4):
        self._circular_max = None
        self._zero_precision = 1e-6 
        self.round_precision = round_precision

    def fit(self, x, y=None):
        if isinstance(x, pd.DataFrame):
            self._circular_max = np.max(np.abs(x.to_numpy()), axis=0)
        else:
            self._circular_max = np.max(np.abs(x), axis=0)
        self._circular_max = np.where(np.abs(self._circular_max) <= self._zero_precision, 1, self._circular_max)
        return self        

    def transform(self, x):
        if self._circular_max is None or not self._circular_max.shape[0]:
            return x
        
        result = []

        if isinstance(x, pd.DataFrame):
            x_ndarray = x.to_numpy()
            columns = []

            for num, col in enumerate(x_ndarray.T):
                result.append(np.sin((2 * np.pi * col) / self._circular_max[num]))
                columns.append(f'sin_{x.columns[num]}')
                result.append(np.cos((2 * np.pi * col) / self._circular_max[num]))
                columns.append(f'cos_{x.columns[num]}')
            return pd.DataFrame(np.round(np.array(result).T, self.round_precision), columns=columns)
        else:
            for num, col in enumerate(x.T):
                result.append(np.sin((2 * np.pi * col) / self._circular_max[num]))
                result.append(np.cos((2 * np.pi * col) / self._circular_max[num]))
            return np.round(np.array(result).T, self.round_precision)


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
    def __init__(self, encoder='standard', numerical_rate=0.2, **encoder_params):
        self._encoders = {'standard': skpr.StandardScaler, 
                          'min_max': skpr.MinMaxScaler, 
                          'normalizer': skpr.Normalizer,
                          'robust': skpr.RobustScaler,
                          'max_abs': skpr.MaxAbsScaler}
        self._encoder_name = encoder
        self._encoder = self._encoders[encoder](**encoder_params)
        self._numerical_columns_ind = None
        self.numerical_rate = numerical_rate

    def fit(self, x, y=None, **fit_params):
        self._numerical_columns_ind = self.get_numerical_columns_inds(x)

        if self._numerical_columns_ind.shape[0] == 0:
            return self        
        x_num = x[:, self._numerical_columns_ind]

        self._encoder.fit(x_num, y, **fit_params)
        return self

    def transform(self, x):
        if self._numerical_columns_ind is None or not self._numerical_columns_ind.shape[0]:
            return x
        x_num_transformed = self._encoder.transform(x[:, self._numerical_columns_ind])
        x_copy = x.copy()
        x_copy[:, self._numerical_columns_ind] = x_num_transformed
        
        return x_copy

    def get_encoder_name(self):
        return self._encoder_name

    def get_available_encoders(self):
        return np.array(list(self._encoders.keys()))

    def get_numerical_columns_inds(self, data):
        numerical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] >= data.shape[0] * self.numerical_rate:
                numerical_features.append(num)
        return np.array(numerical_features)


def main():
    pass
    # data = load_breast_cancer()
    # x = data.data
    # y = data.target
    # x = np.array([[round(col, 1) for col in row] for row in x])


    # a = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['a', 'b', 'c'])
    # print(a.columns[0])
    # a = a.T
    # for ind in a.index:
    #     a.loc[ind] = [1, 0, 0]
    # a = a.T
    # print(a)



    # print('DATAFRAME:')
    # print(a)

    # print(a.iloc[1].to_numpy())
    # num_enc = NumericalEncoder(encoder='max_abs')
    # cat_enc = CategoricalEncoder()

    # x = np.array([[0, 1, 2, 3], [1, 11, 0, 3], [23, 0, 5, 8]]).T
    # circ = CircularEncoder()
    # circ.fit(x)
    # result = circ.transform(x)
    # print(result)

    # print(num_enc.get_numerical_columns_inds(x))
    # print(cat_enc.get_categorical_columns_inds(x))

    # print(x[:, 0])
    # result = num_enc.fit_transform(x, y)
    # result = cat_enc.fit_transform(result, y)
    # print(result[:, 0])




if __name__ == '__main__':
    main()

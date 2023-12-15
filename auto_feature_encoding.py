from sklearn.datasets import load_breast_cancer
import sklearn.preprocessing as skpr
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
import numpy as np
import pandas as pd
import datetime


class CircularEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, limit=None):
        self._circular_max = limit
        self._zero_precision = 1e-6

    def fit(self, X, y=None):
        if self._circular_max is not None:
            if len(X.shape) > 1:
                size = X.shape[1]
            else:
                size = 1
            self._circular_max = np.array([self._circular_max] * size)
        else:
            if isinstance(X, pd.DataFrame):
                self._circular_max = np.max(np.abs(X.to_numpy()), axis=0)
            else:
                self._circular_max = np.max(np.abs(X), axis=0)
            self._circular_max = np.where(np.abs(self._circular_max) <= self._zero_precision, 1, self._circular_max)
        return self

    def transform(self, X):
        if self._circular_max is None or not self._circular_max.shape[0]:
            return X

        columns = None

        if isinstance(X, pd.DataFrame):
            X_ndarray = X.to_numpy()
            columns = np.zeros(2 * X.columns.shape[0], dtype=object)
        else:
            X_ndarray = X.copy()

        if len(X_ndarray.shape) > 1:
            size = len(X_ndarray.shape)
        else:
            size = 1

        result_sin = np.sin((2 * np.pi * X_ndarray) / self._circular_max)
        result_cos = np.cos((2 * np.pi * X_ndarray) / self._circular_max)

        if size == 1:
            result_sin = result_sin.reshape(-1, 1)
            result_cos = result_cos.reshape(-1, 1)

        result = np.zeros((X_ndarray.shape[0], 2 * size))
        result[:, np.arange(0, result.shape[1], 2)] = result_sin
        result[:, np.arange(1, result.shape[1], 2)] = result_cos
        result = np.where(abs(result) < self._zero_precision, 0, result)

        if columns is not None:
            columns[np.arange(0, result.shape[1], 2)] = np.array([f'sin_{col}' for col in X.columns])
            columns[np.arange(1, result.shape[1], 2)] = np.array([f'cos_{col}' for col in X.columns])
            return pd.DataFrame(result, columns=columns)
        else:
            return result


class DateTimeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_inds=None):
        if datetime_inds is not None and not isinstance(datetime_inds, np.ndarray):
            raise TypeError('datetime_inds should be a np.ndarray')
        
        self._datetime_columns_ind = datetime_inds

        self.circular_enc_12 = CircularEncoder(limit=12)
        self.circular_enc_30 = CircularEncoder(limit=30)
        self.circular_enc_60 = CircularEncoder(limit=60)

    def fit(self, X, y=None):
        if self._datetime_columns_ind is None:
            self._datetime_columns_ind = self.get_datetime_columns_inds(X)

        if self._datetime_columns_ind.shape[0] == 0:
            return self

    def transform_column(self, column):
        if not isinstance(column, pd.Series):
            col = pd.Series(column)
        else:
            col = column.copy()

        new_cols = np.array([col.dt.year.to_numpy()]).T
        new_cols = np.concatenate((new_cols, self.circular_enc_12.fit_transform(col.dt.month.to_numpy())), axis=1)
        new_cols = np.concatenate((new_cols, self.circular_enc_30.fit_transform(col.dt.day.to_numpy())), axis=1)
        new_cols = np.concatenate((new_cols, self.circular_enc_60.fit_transform(col.dt.hour.to_numpy())), axis=1)
        new_cols = np.concatenate((new_cols, self.circular_enc_60.fit_transform(col.dt.minute.to_numpy())), axis=1)
        new_cols = np.concatenate((new_cols, self.circular_enc_60.fit_transform(col.dt.second.to_numpy())), axis=1)

        if isinstance(column, pd.Series):
            columns = np.array([f'{column.name}_year',
                                f'{column.name}_month_sin', f'{column.name}_month_cos',
                                f'{column.name}_day_sin', f'{column.name}_day_cos',
                                f'{column.name}_hour_sin', f'{column.name}_hour_cos',
                                f'{column.name}_minute_sin', f'{column.name}_minute_cos',
                                f'{column.name}_second_sin', f'{column.name}_second_cos'])
            new_cols = pd.DataFrame(new_cols, columns=columns)

        return new_cols
    
    def get_datetime_columns_inds(self, data):
        datetime_inds = []
        for num, col in enumerate(data.T):
            is_datetime_st = np.array([self._is_datetime_string(s) for s in col])
            if np.all(np.array([self._is_datetime_string(s) for s in col])):
                datetime_inds.append(num)
        return np.array(datetime_inds)

    def _is_datetime_string(self, s):
        try:
            np.datetime64(s)
            return True
        except ValueError:
            return False        


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder='target_loo', category_rate=0.1, drop=True, cat_inds=None, **encoder_params):
        if cat_inds is not None and not isinstance(cat_inds, np.ndarray):
            raise TypeError('cat_inds should be a np.ndarray')

        self._encoders = {'one_hot': ce.OneHotEncoder,
                          'target_loo': ce.LeaveOneOutEncoder,
                          'hashing': ce.HashingEncoder,
                          'binary': ce.BinaryEncoder}
        self._encoder_name = encoder
        self._encoder_params = encoder_params
        self._encoder = self._encoders[encoder](**encoder_params)
        self._drop = drop
        self._categorical_columns_ind = cat_inds
        self.category_rate = category_rate
        self._encoders_list = None

    def fit(self, X, y=None, **fit_params):
        if self._categorical_columns_ind is None:
            self._categorical_columns_ind = self.get_categorical_columns_inds(X)
        if self._categorical_columns_ind.shape[0] == 0:
            return self
        if isinstance(X, pd.DataFrame):
            X_cat = X.to_numpy()[:, self._categorical_columns_ind]
        else:
            X_cat = X[:, self._categorical_columns_ind]

        self._encoder.fit(pd.DataFrame(X_cat, dtype='category'), y, **fit_params)
        self._encoders_list = np.array([self._encoders[self._encoder_name](**self._encoder_params) for i in range(self._categorical_columns_ind.shape[0])])
        self._encoders_list = np.array([enc.fit(pd.DataFrame(X_cat[:, [i]], dtype='category'), y, **fit_params) for i, enc in enumerate(self._encoders_list)])
        return self

    def transform(self, X):
        if self._categorical_columns_ind is None or not self._categorical_columns_ind.shape[0]:
            return X

        columns = None

        if isinstance(X, pd.DataFrame):
            X_copy = X.to_numpy()
            columns = X.columns.to_numpy()
        else:
            X_copy = X.copy()

        cat_inds = self._categorical_columns_ind.copy()

        for num, cat_ind in enumerate(cat_inds):
            transformed = self._encoders_list[num].transform(pd.DataFrame(X_copy[:, [cat_ind]], dtype='category')).to_numpy()

            if len(transformed.shape) == 1 or transformed.shape[1] == 1:
                if columns is not None:
                    columns[cat_ind] = f'{columns[cat_ind]}_{self._encoder_name}'
                    X_copy[:, [cat_ind]] = transformed
            else:
                if columns is not None:
                    extra_names = np.array([f'{columns[cat_ind]}_{self._encoder_name}_{ind}' for ind in np.arange(transformed.shape[1])])
                    columns = np.concatenate((columns[:cat_ind + 1], extra_names, columns[cat_ind + 1:]))
                X_copy = np.concatenate((X_copy[:, :cat_ind + 1], transformed, X_copy[:, cat_ind + 1:]), axis=1)

                cat_inds[num + 1:] += transformed.shape[1]

                if self._drop:
                    if columns is not None:
                        columns = np.delete(columns, [cat_ind])
                    X_copy = np.delete(X_copy, [cat_ind], axis=1)
                    cat_inds[num + 1:] -= 1

        if columns is None:
            return X_copy
        else:
            return pd.DataFrame(X_copy, columns=columns)

    def get_encoder_name(self):
        return self._encoder_name

    def get_available_encoders(self):
        return np.array(list(self._encoders.keys()))

    def get_categorical_columns_inds(self, data):
        if isinstance(data, pd.DataFrame):
            suitable_dtypes = ['category', 'object']
            columns = data.select_dtypes(include=suitable_dtypes).columns.to_numpy()
            all_columns = data.columns.to_numpy()
            return np.array([np.where(all_columns == col)[0] for col in columns]).reshape(-1)

        categorical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] < data.shape[0] * self.category_rate:
                categorical_features.append(num)
        return np.array(categorical_features)


class NumericalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder='standard', numerical_rate=0.1, num_inds=None, **encoder_params):
        if num_inds is not None and not isinstance(num_inds, np.ndarray):
            raise TypeError('num_inds should be a np.ndarray')
        self._encoders = {'standard': skpr.StandardScaler,
                          'min_max': skpr.MinMaxScaler,
                          'normalizer': skpr.Normalizer,
                          'max_abs': skpr.MaxAbsScaler}
        self._encoder_name = encoder
        self._encoder = self._encoders[encoder](**encoder_params)
        self._numerical_columns_ind = num_inds
        self.numerical_rate = numerical_rate

    def fit(self, X, y=None, **fit_params):
        if self._numerical_columns_ind is None:
            self._numerical_columns_ind = self.get_numerical_columns_inds(X)

        if self._numerical_columns_ind.shape[0] == 0:
            return self

        if isinstance(X, pd.DataFrame):
            X_num = X.to_numpy()[:, self._numerical_columns_ind]
        else:
            X_num = X[:, self._numerical_columns_ind]

        self._encoder.fit(X_num, y, **fit_params)
        return self

    def transform(self, X):
        if self._numerical_columns_ind is None or not self._numerical_columns_ind.shape[0]:
            return X

        columns = None

        if isinstance(X, pd.DataFrame):
            X_copy = X.to_numpy().astype(float)
            columns = X.columns.to_numpy()
        else:
            X_copy = X.copy().astype(float)

        if columns is not None:
            columns[self._numerical_columns_ind] = np.array([f'{name}_{self._encoder_name}' for name in columns[self._numerical_columns_ind]])

        X_num_transformed = self._encoder.transform(X_copy[:, self._numerical_columns_ind])
        X_copy[:, self._numerical_columns_ind] = X_num_transformed

        if columns is None:
            return X_copy
        else:
            return pd.DataFrame(X_copy, columns=columns)

    def get_encoder_name(self):
        return self._encoder_name

    def get_available_encoders(self):
        return np.array(list(self._encoders.keys()))

    def get_numerical_columns_inds(self, data):
        if isinstance(data, pd.DataFrame):
            wrong_dtypes = ['object', 'category', 'datetime64', 'timedelta']
            columns = data.select_dtypes(exclude=wrong_dtypes).columns.to_numpy()
            all_columns = data.columns.to_numpy()
            return np.array([np.where(all_columns == col)[0] for col in columns]).reshape(-1)

        numerical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] >= data.shape[0] * self.numerical_rate:
                numerical_features.append(num)
        return np.array(numerical_features)


class BasicEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, category_params, numerical_params):
        # category_params - dict
        # numerical_params - dict

        self.datetime_enc = DateTimeEncoder()
        self.category_enc = CategoricalEncoder(**category_params)
        self.numerical_enc = NumericalEncoder(**numerical_params)
    
    def fit(self, X, y, **fit_params):
        pass

    def transformed(self, X):
        pass
    
    def get_columns_classification(self):
        pass
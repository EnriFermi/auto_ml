from sklearn.datasets import load_breast_cancer
from IPython.display import display 
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


class Encoder:
    def __init__(self, enc_type):
        self.enc_type = enc_type
        self.encode_functions = {'one_hot': self._one_hot_enc}
        self.category_rate = 0.2

    def encode(self, data):
        encoded_data = None
        if self.enc_type in self.encode_functions.keys():
            encoded_data = self.encode_functions[self.enc_type](data)
        
        return encoded_data

    def _encode(self, data, features_inds, enc):
        for item in features_inds:
            encoded = enc.fit_transform(data[:, item].reshape(-1, 1)).toarray()
            data = np.delete(data, item, 1) 
            data = np.concatenate((data, encoded), axis=1)
        return data

    def _get_categorical_inds(self, data):
        categorical_features = list()

        for num, col in enumerate(data.T):
            if np.unique(col).shape[0] < data.shape[0] * self.category_rate:
                categorical_features.append(num)
        return categorical_features

    def _one_hot_enc(self, data):
        enc = OneHotEncoder()
        categorical_features = self._get_categorical_inds(data)
        return self._encode(data, categorical_features, enc)


def main():
    data = load_breast_cancer()
    x = data.data
    y = data.target

    for i in range(x.shape[0]):
        x[i] = np.array(list(map(int, x[i]))) # Example dataset, turn real features into categories

    enc = Encoder("one_hot")
    result = enc.encode(x)
    print(pd.DataFrame(result, dtype='float32').head())


if __name__ == '__main__':
    main()

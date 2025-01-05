class ValidTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        self.kwargs = kwargs
        if 'work_time' in kwargs.keys():
            self.work_time = kwargs['work_time']
        else:
            self.work_time = 1
        if 'val_size' in kwargs.keys():
            self.val_size = kwargs['val_size']
        else:
            self.val_size = 0.2
        if 'random_state' in kwargs.keys():
            self.random_state = kwargs['random_state']
        else:
            self.random_state = 42

    def fit(self, X, y=None):
        obj_count, f_dim = X.shape[0], X.shape[1]

        if 'encoder' in self.kwargs.keys():
            self.encoder = CategoricalEncoder(**self.kwargs['encoder'])
        else:
            self.encoder = CategoricalEncoder()
        # self.encoder = preprocessing.StandardScaler() # TODO add normalization after generation
        if 'processor' in self.kwargs.keys():
            self.trans = FeatProc(**self.kwargs['processor'])
        else:
            self.trans = FeatProc(self.task_type, 2)
        self.encoder.fit(X, y)
        self.trans.fit(X, y)

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X = self.encoder.transform(X)
        return self.trans.transform(X)

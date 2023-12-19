def merge_desc_dict(dict1, dict2, bias=0):
    rdict = {}
    pointer = bias
    for op in dict1.keys():
        val = np.unique(np.concatenate(
            (dict1[op][0], dict2[op][0]), axis=0), axis=0)
        rdict[op] = [val, np.arange(pointer, pointer+val.shape[0])]
        pointer += val.shape[0]
    return rdict


class FeatProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, task_type, n_fold_splits, gen_kwargs={}, sel_for_gen_kwargs={}, sel_from_gen_kwargs={}):
        self.n_fold_splits = n_fold_splits
        self.gen_kwargs = gen_kwargs
        self.sel_for_gen_kwargs = sel_for_gen_kwargs
        self.sel_from_gen_kwargs = sel_from_gen_kwargs

        self.desc_dict = {}

    def fit(self, X, y=None):
        print(X.shape)

        if 'k' not in self.sel_for_gen_kwargs.keys():
            for_k = X.shape[1] // 2
            self.sel_for_gen_kwargs['k'] = for_k
        else:
            for_k = self.sel_for_gen_kwargs['k']

        if 'k' not in self.sel_from_gen_kwargs.keys():
            from_k = X.shape[1]
            self.sel_for_gen_kwargs['k'] = from_k
        else:
            from_k = self.sel_from_gen_kwargs['k']

        kf = KFold(n_splits=self.n_fold_splits)
        for i, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val, y_train, y_val = X[train_index,
                                               :], X[val_index, :], y[train_index], y[val_index]
            obj_count, f_dim = X_train.shape[0], X_train.shape[1]
            self.sel_for_gen = FeatureSelectionTransformer(
                **self.sel_for_gen_kwargs)
            self.sel_for_gen.fit(X_train, y_train)
            important_features = self.sel_for_gen.get_support(indices=True)[
                :for_k]
            self.gen = FeatureGenerationTransformer(
                features_mask=important_features, important_features=important_features, **self.gen_kwargs)
            self.gen.fit(X_train, y_train)
            X_gen = self.gen.transform(X_val)

            print(X_gen.shape, X_train.shape, important_features.shape)
            self.sel_from_gen = FeatureSelectionTransformer(
                **self.sel_from_gen_kwargs)
            self.sel_from_gen.fit(X_gen, y_val)

            frs = self.sel_from_gen.get_support(indices=True)[:from_k]
            fos = important_features

            desc = self.gen.desc_dict
            desc['_'] = [np.arange(0, X.shape[1]), np.arange(0, X.shape[1])]
            sort_desc = {}
            map = np.zeros(X_gen.shape[0])

            # for keys
            for op in desc.keys():
               # Начинается с 0
                # remove generating which are not selected
                mask = np.in1d(desc[op][1], frs)
                print(frs.shape, fos.shape)
                sort_desc[op] = [desc[op][0][mask], desc[op][1][mask]]
                if op != '_':
                    sort_desc[op][0] = fos[sort_desc[op][0]]
            print(sort_desc)
            if len(self.desc_dict.keys()) == 0:
                self.desc_dict = sort_desc
            else:
                self.desc_dict = merge_desc_dict(
                    sort_desc, self.desc_dict)  # union algorythms

        self.gen.desc_dict = self.desc_dict  # !!!!!

    def transform(self, X):

        return self.gen.transform(X)

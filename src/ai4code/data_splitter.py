from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

def create_binary_split(df, df_ancestors, test_ratio):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=0)

    # Split, keeping notebooks with a common origin (ancestor_id) together
    ids = df.index.unique('id')
    ancestors = df_ancestors.loc[ids, 'ancestor_id']
    ids_train, ids_valid = next(splitter.split(ids, groups=ancestors))
    ids_train, ids_valid = ids[ids_train], ids[ids_valid]

    fold_idx = pd.Series(index=df.index, dtype='int64')
    fold_idx[ids_train] = 1
    fold_idx[ids_valid] = 0

    to_train = pd.Series(index=df.index, dtype='int64')
    to_train[ids_train] = 1
    to_train[ids_valid] = 0

    return fold_idx, to_train

#self.df_train, self.df_valid, self.df_ranks_train, self.df_ranks_valid = self._create_split(data_dir, df, df_ranks)


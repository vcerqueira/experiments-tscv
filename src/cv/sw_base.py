from abc import ABC

import numpy as np
import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator


class SeriesWiseTimeSeriesCV(BaseCrossValidator, ABC):

    def __init__(self, n_splits=None, id_col='unique_id', time_col: str = 'ds'):
        self.n_splits = n_splits
        self.id_col = id_col
        self.time_col = time_col

    def split(self, X, y=None, groups=None):
        pass

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def time_wise_split(self, df: pd.DataFrame, horizon: int):
        df_by_unq = df.groupby(self.id_col)

        train_l, test_l = [], []
        for g, df_ in df_by_unq:
            df_ = df_.sort_values(self.time_col)

            train_df_g = df_.head(-horizon)
            test_df_g = df_.tail(horizon)

            train_l.append(train_df_g)
            test_l.append(test_df_g)

        train_df = pd.concat(train_l).reset_index(drop=True)
        test_df = pd.concat(test_l).reset_index(drop=True)

        return train_df, test_df

    def get_sets_from_idx(self, df: pd.DataFrame, uids: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
        train_uids, test_uids = uids[train_idx], uids[test_idx]

        # reflect repetitions in train_idx (for bootstrapping)
        train_df = pd.concat(
            [df[df[self.id_col] == uid] for uid in train_uids],
            ignore_index=True
        )

        test_df = pd.concat(
            [df[df[self.id_col] == uid] for uid in test_uids],
            ignore_index=True
        )

        return train_df, test_df

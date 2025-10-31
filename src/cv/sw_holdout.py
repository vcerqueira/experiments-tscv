from abc import ABC

import numpy as np

from src.cv.sw_base import SeriesWiseTimeSeriesCV


class SeriesWiseHoldout(SeriesWiseTimeSeriesCV, ABC):

    def __init__(self, train_size: float, id_col: str = 'unique_id'):
        self.train_size = train_size

        super().__init__(n_splits=1, id_col=id_col)

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_train = int(n_samples * self.train_size)

        indices = np.arange(n_samples)
        train_idx = np.random.choice(indices, size=n_train, replace=False)
        test_idx = np.setdiff1d(indices, train_idx)

        yield train_idx, test_idx

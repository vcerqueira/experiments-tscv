from abc import ABC
from typing import Optional

import numpy as np

from src.cv.sw_base import SeriesWiseTimeSeriesCV


class SeriesWiseBootstrap(SeriesWiseTimeSeriesCV, ABC):

    def __init__(self,
                 random_state: Optional[int] = None,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds'):
        self.random_state = random_state
        super().__init__(n_splits=1, id_col=id_col, time_col=time_col)

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)

        indices = np.arange(n_samples)
        train_idx = rng.choice(indices, size=n_samples, replace=True)
        test_idx = np.setdiff1d(indices, np.unique(train_idx))

        yield train_idx, test_idx


class SeriesWiseRepeatedBootstrap(SeriesWiseTimeSeriesCV, ABC):

    def __init__(self,
                 n_repeats: int,
                 random_state: Optional[int] = None,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds'):
        self.n_repeats = n_repeats
        self.random_state = random_state
        super().__init__(n_splits=n_repeats, id_col=id_col, time_col=time_col)

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        indices = np.arange(n_samples)

        for _ in range(self.n_repeats):
            # Bootstrapping: sample with replacement, as in SeriesWiseBootstrap
            train_idx = rng.choice(indices, size=n_samples, replace=True)
            test_idx = np.setdiff1d(indices, np.unique(train_idx))
            yield train_idx, test_idx

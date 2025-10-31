from abc import ABC

import numpy as np

from src.cv.sw_base import SeriesWiseTimeSeriesCV


class SeriesWiseHoldout(SeriesWiseTimeSeriesCV, ABC):

    def __init__(self,
                 train_size: float,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds'):
        self.train_size = train_size

        super().__init__(n_splits=1, id_col=id_col, time_col=time_col)

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_train = int(n_samples * self.train_size)

        indices = np.arange(n_samples)
        train_idx = np.random.choice(indices, size=n_train, replace=False)
        test_idx = np.setdiff1d(indices, train_idx)

        yield train_idx, test_idx


class SeriesWiseRepeatedHoldout(SeriesWiseTimeSeriesCV, ABC):

    def __init__(self,
                 train_size: float,
                 n_repeats: int,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds'):
        self.train_size = train_size
        self.n_repeats = n_repeats
        super().__init__(n_splits=n_repeats, id_col=id_col, time_col=time_col)

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_train = int(n_samples * self.train_size)
        indices = np.arange(n_samples)

        for _ in range(self.n_repeats):
            train_idx = np.random.choice(indices, size=n_train, replace=False)
            test_idx = np.setdiff1d(indices, train_idx)
            yield train_idx, test_idx


class SeriesWiseMonteCarlo(SeriesWiseTimeSeriesCV, ABC):

    def __init__(self,
                 train_size: float,
                 test_size: float,
                 n_repeats: int,
                 id_col: str = 'unique_id',
                 time_col: str = 'ds'):

        if train_size + test_size >= 1.0:
            raise ValueError("train_size + test_size must be less than 1.0")

        self.train_size = train_size
        self.test_size = test_size
        self.n_repeats = n_repeats
        super().__init__(n_splits=n_repeats, id_col=id_col, time_col=time_col)

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_train = int(n_samples * self.train_size)
        n_test = int(n_samples * self.test_size)
        indices = np.arange(n_samples)

        for _ in range(self.n_repeats):
            selected_indices = np.random.choice(indices, size=n_train + n_test, replace=False)
            train_idx = np.random.choice(selected_indices, size=n_train, replace=False)
            train_idx_set = set(train_idx)
            test_idx = np.array([idx for idx in selected_indices if idx not in train_idx_set])
            yield train_idx, test_idx

from abc import ABC

from sklearn.model_selection import KFold, RepeatedKFold

from src.cv.sw_base import SeriesWiseTimeSeriesCV


class SeriesWiseKFold(SeriesWiseTimeSeriesCV, ABC):
    def __init__(self, n_splits=5, random_state=None, id_col: str = 'unique_id', time_col: str = 'ds'):
        super().__init__(n_splits=n_splits, id_col=id_col, time_col=time_col)

        self._kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split(self, X, y=None, groups=None):
        yield from self._kf.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._kf.get_n_splits(X, y, groups)


class SeriesWiseRepeatedKFold(SeriesWiseTimeSeriesCV, ABC):
    def __init__(self, n_splits=5, n_repeats=2, random_state=None, id_col: str = 'unique_id', time_col: str = 'ds'):
        super().__init__(n_splits=n_splits, id_col=id_col, time_col=time_col)

        self._rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    def split(self, X, y=None, groups=None):
        yield from self._rkf.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._rkf.get_n_splits(X, y, groups)

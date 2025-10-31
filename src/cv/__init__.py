from src.cv.sw_holdout import (SeriesWiseHoldout,
                               SeriesWiseRepeatedHoldout,
                               SeriesWiseMonteCarlo)
from src.cv.sw_bootstrap import SeriesWiseBootstrap, SeriesWiseRepeatedBootstrap
from sklearn.model_selection import KFold, RepeatedKFold

from src.config import N_FOLDS, HOLDOUT_TR, MC_TR, MC_TS, KFOLD_N_REPEATS

CV_METHODS = {
    'Holdout': SeriesWiseHoldout,
    'RepeatedHoldout': SeriesWiseRepeatedHoldout,
    'MonteCarlo': SeriesWiseMonteCarlo,
    'Bootstrap': SeriesWiseBootstrap,
    'RepeatedBootstrap': SeriesWiseRepeatedBootstrap,
    'KFold': KFold,
    'RepeatedKFold': RepeatedKFold,
}

CV_METHODS_PARAMS = {
    'Holdout': {'train_size': HOLDOUT_TR},
    'RepeatedHoldout': {'train_size': HOLDOUT_TR, 'n_repeats': N_FOLDS},
    'MonteCarlo': {'train_size': MC_TR, 'test_size': MC_TS, 'n_repeats': N_FOLDS},
    'Bootstrap': {},
    'RepeatedBootstrap': {'n_repeats': N_FOLDS},
    'KFold': {'n_splits': N_FOLDS, 'shuffle': True},
    'RepeatedKFold': {'n_splits': N_FOLDS, 'n_repeats': KFOLD_N_REPEATS},
}

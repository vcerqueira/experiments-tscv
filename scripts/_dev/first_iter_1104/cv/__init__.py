from src.cv.sw_holdout import (SeriesWiseHoldout,
                               SeriesWiseRepeatedHoldout,
                               SeriesWiseMonteCarlo)
from src.cv.sw_bootstrap import SeriesWiseBootstrap, SeriesWiseRepeatedBootstrap
from src.cv.sw_kfold import SeriesWiseKFold, SeriesWiseRepeatedKFold

from src.config import N_FOLDS, HOLDOUT_TR, MC_TR, MC_TS, KFOLD_N_REPEATS, SEED

CV_METHODS = {
    'Holdout': SeriesWiseHoldout,
    'RepeatedHoldout': SeriesWiseRepeatedHoldout,
    'MonteCarlo': SeriesWiseMonteCarlo,
    'Bootstrap': SeriesWiseBootstrap,
    # 'RepeatedBootstrap': SeriesWiseRepeatedBootstrap,
    'KFold': SeriesWiseKFold,
    # 'RepeatedKFold': SeriesWiseRepeatedKFold,
}

CV_METHODS_PARAMS = {
    'Holdout': {'train_size': HOLDOUT_TR},
    'RepeatedHoldout': {'train_size': HOLDOUT_TR, 'n_repeats': N_FOLDS},
    'MonteCarlo': {'train_size': MC_TR, 'test_size': MC_TS, 'n_repeats': N_FOLDS},
    'Bootstrap': {},
    'RepeatedBootstrap': {'n_repeats': N_FOLDS},
    'KFold': {'n_splits': N_FOLDS, 'random_state': SEED},
    'RepeatedKFold': {'n_splits': N_FOLDS, 'n_repeats': KFOLD_N_REPEATS, 'random_state': SEED},
}

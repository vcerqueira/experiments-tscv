import copy

import numpy as np
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.neuralnets import ModelsConfig
from src.chronos_data import ChronosDataset
from src.config import SEED


def time_wise_holdout(train, test, models, freq, freq_int, horizon):
    # -- setup
    models_ = copy.deepcopy(models)
    nf = NeuralForecast(models=models_, freq=freq)
    sf_inner = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq, n_jobs=1)

    # -- inner cv,
    nf_inner_setup = {
        'df': train, 'val_size': horizon,
        'test_size': horizon, 'step_size': 1, 'n_windows': None,
    }

    sf_inner_setup = {
        'df': train, 'test_size': horizon,
        'step_size': 1, 'n_windows': None, 'h': horizon
    }

    np.random.seed(SEED)
    cv_inner = nf.cross_validation(**nf_inner_setup)
    cv_inner['fold'] = 0
    cv_sf_inner = sf_inner.cross_validation(**sf_inner_setup)
    cv_inner = cv_inner.merge(cv_sf_inner.drop(columns=['y', 'cutoff']), on=['ds', 'unique_id'], how='left')

    # -- cv "inference"
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_final = NeuralForecast(models=optim_models, freq=freq)

    complete_df = ChronosDataset.concat_time_wise_tr_ts(train, test)

    nf_outer_setup = {
        'df': complete_df, 'val_size': horizon,
        'test_size': horizon * 3, 'step_size': 1, 'n_windows': None
    }

    sf_outer_setup = {
        'df': complete_df, 'h': horizon,
        'test_size': horizon * 3, 'step_size': 1, 'n_windows': None
    }

    cv_nf_f = nf_final.cross_validation(**nf_outer_setup)

    sf = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq)

    cv_sf_f = sf.cross_validation(**sf_outer_setup)

    cv_outer = cv_nf_f.merge(cv_sf_f.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])

    return cv_outer, cv_inner

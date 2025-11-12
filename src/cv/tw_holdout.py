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
    cv_params = {'val_size': horizon, 'test_size': None, 'n_windows': horizon, 'step_size': 1}
    np.random.seed(SEED)
    cv_inner = nf.cross_validation(df=train, **cv_params)
    cv_inner['fold'] = 0
    cv_sf_inner = sf_inner.cross_validation(df=train, h=horizon, test_size=horizon, n_windows=None)
    cv_inner = cv_inner.merge(cv_sf_inner.drop(columns=['y', 'cutoff']), on=['ds', 'unique_id'], how='left')

    # -- cv "inference"
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_final = NeuralForecast(models=optim_models, freq=freq)

    complete_df = ChronosDataset.concat_time_wise_tr_ts(train, test)

    cv_nf_f = nf_final.cross_validation(df=complete_df,
                                        val_size=horizon,
                                        test_size=horizon * 3,
                                        step_size=1,
                                        n_windows=None)

    # cv = fcst.merge(estimation_test, on=['ds', 'unique_id'], how='right')

    sf = StatsForecast(
        models=[SeasonalNaive(season_length=freq_int)],
        freq=freq,
        n_jobs=1)

    cv_sf_f = sf.cross_validation(df=complete_df,
                                  h=horizon,
                                  test_size=horizon * 3,
                                  step_size=1,
                                  n_windows=None)

    cv_outer = cv_nf_f.merge(cv_sf_f.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])

    return cv_outer, cv_inner

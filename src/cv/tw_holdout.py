import copy

import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.neuralnets import ModelsConfig
from src.load_data.base import LoadDataset
from src.config import SEED


def time_wise_holdout2(train, test, models, freq, freq_int, horizon):
    # -- setup
    models_ = copy.deepcopy(models)
    nf = NeuralForecast(models=models_, freq=freq)
    sf_inner = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq, n_jobs=1)

    # -- inner cv
    # cv_params = {'val_size': horizon, 'test_size': horizon, 'n_windows': None, }
    # cv_inner = nf.cross_validation(df=train, **cv_params)
    # cv_inner['fold'] = 0
    # cv_sf_inner = sf_inner.cross_validation(df=train, h=horizon, test_size=horizon, n_windows=None)
    # cv_inner = cv_inner.merge(cv_sf_inner.drop(columns=['y', 'cutoff']), on=['ds', 'unique_id'], how='left')

    dev, valid = LoadDataset.time_wise_split(train, horizon=horizon)
    np.random.seed(SEED)
    nf.fit(df=dev, val_size=horizon)
    sf_inner.fit(df=dev)
    fcst_outsample = nf.predict(df=dev)
    sf_fcst_outsample = sf_inner.predict(h=horizon)
    cv_inner = fcst_outsample.merge(valid, on=['ds', 'unique_id'], how='right')
    cv_inner = cv_inner.merge(sf_fcst_outsample, on=['ds', 'unique_id'], how='left')
    cv_inner['fold'] = 0

    # -- cv "inference"
    optim_models = ModelsConfig.get_best_configs(nf)
    nf_rt = NeuralForecast(models=optim_models, freq=freq)
    nf_rt.fit(df=train, val_size=horizon)
    fcst_rt = nf_rt.predict(df=train)

    # -- merge with test
    cv_rt = fcst_rt.merge(test, on=['ds', 'unique_id'], how='right')

    sf_outer = StatsForecast(
        models=[SeasonalNaive(season_length=freq_int)],
        freq=freq,
        n_jobs=1)

    sf_outer.fit(df=train)
    fcst_sf = sf_outer.predict(h=horizon)

    cv_rt = fcst_sf.merge(cv_rt, on=['ds', 'unique_id'], how='right')

    return cv_rt, cv_inner


def time_wise_holdout(train, test, complete_df, models, freq, freq_int, horizon):
    # -- setup
    models_ = copy.deepcopy(models)
    nf = NeuralForecast(models=models_, freq=freq)
    sf_inner = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq, n_jobs=1)

    # -- inner cv,
    cv_params = {'val_size': horizon, 'test_size': None, 'n_windows': horizon, 'step_size': 1}
    cv_inner = nf.cross_validation(df=train, **cv_params)
    cv_inner['fold'] = 0
    cv_sf_inner = sf_inner.cross_validation(df=train, h=horizon, test_size=horizon, n_windows=None)
    cv_inner = cv_inner.merge(cv_sf_inner.drop(columns=['y', 'cutoff']), on=['ds', 'unique_id'], how='left')

    # -- cv "inference"
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_final = NeuralForecast(models=optim_models, freq=freq)
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

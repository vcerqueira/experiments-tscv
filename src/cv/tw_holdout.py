import copy

from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.neuralnets import ModelsConfig


def time_wise_holdout(train, test, models, freq, freq_int, horizon):
    # -- setup
    models_ = copy.deepcopy(models)
    nf = NeuralForecast(models=models_, freq=freq)
    sf_inner = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq, n_jobs=1)

    # -- inner cv
    cv_params = {'val_size': horizon, 'test_size': horizon, 'n_windows': None, }
    cv_inner = nf.cross_validation(df=train, **cv_params)
    cv_inner['fold'] = 0
    cv_sf_inner = sf_inner.cross_validation(df=train, h=horizon, test_size=horizon, n_windows=None)
    cv_inner = cv_inner.merge(cv_sf_inner.drop(columns=['y','cutoff']), on=['ds', 'unique_id'],how='left')

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

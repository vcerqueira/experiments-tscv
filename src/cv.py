from neuralforecast import NeuralForecast

from src.neuralnets import ModelsConfig


def time_wise_holdout(train, test, models, freq, horizon):
    # -- setup
    nf = NeuralForecast(models=models, freq=freq)

    # -- inner cv
    cv_params = {'val_size': horizon, 'test_size': horizon, 'n_windows': None, }
    cv_inner = nf.cross_validation(df=train, **cv_params)

    # -- get best config; easy with time-wise holdout. how to with series-wise?
    optim_models = ModelsConfig.get_best_configs(nf)

    # -- "inference"
    fcst = nf.predict(df=train)

    nf_rt = NeuralForecast(models=optim_models, freq=freq)
    nf_rt.fit(df=train, val_size=horizon)
    fcst_rt = nf_rt.predict(df=train)

    # -- merge with test
    cv = fcst.merge(test, on=['ds', 'unique_id'], how='right')
    cv_rt = fcst_rt.merge(test, on=['ds', 'unique_id'], how='right')

    return cv_inner, cv, cv_rt



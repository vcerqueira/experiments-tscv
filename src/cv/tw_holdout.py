from neuralforecast import NeuralForecast

from src.neuralnets import ModelsConfig


def time_wise_holdout(train, test, models, freq, horizon):
    # -- setup
    nf = NeuralForecast(models=models, freq=freq)

    # -- inner cv
    cv_params = {'val_size': horizon, 'test_size': horizon, 'n_windows': None, }
    cv_inner = nf.cross_validation(df=train, **cv_params)

    # -- "inference"
    fcst = nf.predict(df=train)

    # -- get best config; easy with time-wise holdout. how to with series-wise?
    optim_models = ModelsConfig.get_best_configs(nf)
    nf_rt = NeuralForecast(models=optim_models, freq=freq)
    nf_rt.fit(df=train, val_size=horizon)
    fcst_rt = nf_rt.predict(df=train)

    # -- merge with test
    cv = fcst.merge(test, on=['ds', 'unique_id'], how='right')
    cv_rt = fcst_rt.merge(test, on=['ds', 'unique_id'], how='right')

    return cv_inner, cv, cv_rt

#
# def series_wise_holdout(df: pd.DataFrame,
#                         train_size: float,
#                         id_col: str = 'unique_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
#     uids = df[id_col].unique()
#     n_train = int(len(uids) * train_size)
#
#     train_ids = np.random.choice(uids, size=n_train, replace=False)
#
#     is_train_obs = df[id_col].isin(train_ids)
#
#     train_df = df[is_train_obs].reset_index(drop=True)
#     test_df = df[~is_train_obs].reset_index(drop=True)
#
#     return train_df, test_df

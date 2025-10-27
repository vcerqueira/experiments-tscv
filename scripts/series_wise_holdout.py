import os
import warnings

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig
from src.config import N_SAMPLES, RETRAIN_FOR_TEST

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
DRY_RUN = True
GROUP_IDX = 0
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

n_uids, n_trials, retrain = (30, 2, False) if DRY_RUN else (None, N_SAMPLES, RETRAIN_FOR_TEST)

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=n_uids)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

estimation_train, estimation_test = data_loader.time_wise_split(df, horizon=horizon)

models = ModelsConfig.get_auto_nf_models(horizon=horizon, n_samples=n_trials)

# ---- model setup
nf = NeuralForecast(models=models, freq=freq_str)

uids = estimation_train['unique_id'].unique()
n_train = int(len(uids) * 0.7)
train_ids = np.random.choice(uids, size=n_train, replace=False)

is_train_obs = estimation_train['unique_id'].isin(train_ids)

train_df = estimation_train[is_train_obs].reset_index(drop=True)
test_df = estimation_train[~is_train_obs].reset_index(drop=True)

from typing import Tuple


def series_wise_holdout(df: pd.DataFrame,
                        train_size: float,
                        id_col: str = 'unique_id') -> Tuple[pd.DataFrame, pd.DataFrame]:
    uids = df[id_col].unique()
    n_train = int(len(uids) * train_size)

    train_ids = np.random.choice(uids, size=n_train, replace=False)

    is_train_obs = df[id_col].isin(train_ids)

    train_df = df[is_train_obs].reset_index(drop=True)
    test_df = df[~is_train_obs].reset_index(drop=True)

    return train_df, test_df


series_wise_holdout(estimation_train, train_size=0.5)


cv = nf.cross_validation(df=estimation_train, val_size=horizon, test_size=horizon, n_windows=None)

fcst = nf.predict(df=estimation_train)

if retrain:
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_rt = NeuralForecast(models=optim_models, freq=freq_str)
    nf_rt.fit(df=estimation_train, val_size=horizon)
    fcst_rt = nf_rt.predict(df=estimation_train)

print(fcst)

cv = fcst.merge(estimation_test, on=['ds', 'unique_id'], how='right')

# cv.to_csv(f'assets/results/{data_name},{group},dry-run.csv', index=True)

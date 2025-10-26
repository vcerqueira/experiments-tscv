import os
import warnings

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

import numpy as np

# Get unique IDs and randomly split into train/test
unique_ids = estimation_train['unique_id'].unique()
n_train = int(len(unique_ids) * 0.7)
train_ids = np.random.choice(unique_ids, size=n_train, replace=False)
test_ids = np.array(list(set(unique_ids) - set(train_ids)))

# Split data based on IDs
train_df = estimation_train[estimation_train['unique_id'].isin(train_ids)]
test_df = estimation_train[estimation_train['unique_id'].isin(test_ids)]


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

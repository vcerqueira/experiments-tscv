import os
import copy
import warnings

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from sklearn.model_selection import KFold

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig
from src.config import N_SAMPLES, N_FOLDS, SEED
from src.neuralforecast_ext import NeuralForecast2

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
DRY_RUN = True
GROUP_IDX = 0
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=30)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
estimation_train, estimation_test = data_loader.time_wise_split(df, horizon=horizon)

# ---- model setup
models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                         try_mps=False,
                                         limit_epochs=True,
                                         n_samples=2)[:2]

nf = NeuralForecast(models=models, freq=freq_str)

# note that this is cv on the time series set (80% of time series for train, 20% for testing)
# partition is done at time series level, not in time dimension
kfcv = KFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)

uids = df['unique_id'].unique()

results, folds_scores = [], []
for j, (train_index, test_index) in enumerate(kfcv.split(uids)):
    if j > 1:
        continue
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    train_uids, test_uids = uids[train_index], uids[test_index]

    models_ = copy.deepcopy(models)
    nf = NeuralForecast2(models=models_, freq=freq_str, train_uids=train_uids)
    np.random.seed(123)
    cv_nf = nf.cross_validation(df=estimation_train,
                                val_size=horizon,
                                test_size=None,
                                step_size=1,
                                n_windows=horizon)

    valid_scores = ModelsConfig.get_all_config_results(nf)
    folds_scores.append(valid_scores)

optim_models = ModelsConfig.get_best_configs(folds_scores)

folds_fl = [item for sublist in folds_scores for item in sublist]

folds_df = pd.DataFrame(folds_fl)

folds_avg = folds_df.groupby(['model', 'hash_value']).mean(numeric_only=True)

best_configs = folds_avg.loc[folds_avg.groupby('model')['loss'].idxmin()].reset_index()

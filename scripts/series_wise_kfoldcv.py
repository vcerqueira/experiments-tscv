import os
import warnings
import hashlib
import json

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from sklearn.model_selection import KFold

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig
from src.config import N_SAMPLES, RETRAIN_FOR_TEST, N_FOLDS
from src.cv import series_wise_holdout

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

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
estimation_train, estimation_test = data_loader.time_wise_split(df, horizon=horizon)

# ---- model setup
models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                         try_mps=False,
                                         limit_epochs=True,
                                         n_samples=n_trials)

nf = NeuralForecast(models=models, freq=freq_str)

# note that this is cv on the time series set (80% of time series for train, 20% for testing)
# partition is done at time series level, not in time dimension
kfcv = KFold(n_splits=N_FOLDS, random_state=123, shuffle=True)

uids = df['unique_id'].unique()


# is_train_obs = df[id_col].isin(train_ids)
# train_df = df[is_train_obs].reset_index(drop=True)
# test_df = df[~is_train_obs].reset_index(drop=True)


def get_all_config_results(nf: NeuralForecast):
    # nf.models[0].results[0].metrics['loss']
    # nf.models[0].results[0].metrics['config']

    scores = []
    for mod in nf.models:
        print(f"Model: {mod.alias}")
        for i, res in enumerate(mod.results):
            print(i)
            res.config['learning_rate'] = np.round(res.config['learning_rate'], 5)

            conf_str = {k: str(v) for k, v in res.config.items()}
            sorted_string = json.dumps(conf_str, sort_keys=True)
            hash_value = hashlib.md5(sorted_string.encode()).hexdigest()

            scores.append({
                'model': mod.alias,
                'config_idx': i,
                'loss': res.metrics['loss'],
                'config': res.config,
                'hash_value': hash_value
            })

    return scores


scores_folds = []
results = []
for j, (train_index, test_index) in enumerate(kfcv.split(uids)):
    if j > 1:
        continue
    print(f"Fold {j}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

    train_uids = uids[train_index]
    is_train_obs = df['unique_id'].isin(train_uids)

    train = df[is_train_obs].reset_index(drop=True)
    test = df[~is_train_obs].reset_index(drop=True)

    # --- inner cv setup
    # --- we split train and test further by time to get insample and outsample parts
    # --- training is done using train_insample; train_outsample is not used.
    # --- testing is done using test_outsample; test_insample is used for generating forecasts
    train_insample, _ = data_loader.time_wise_split(train, horizon=horizon)
    test_insample, test_outsample = data_loader.time_wise_split(test, horizon=horizon)

    np.random.seed(123)
    nf.fit(df=train_insample, val_size=horizon)

    fcst_outsample = nf.predict(df=test_insample)

    cv_inner = fcst_outsample.merge(test_outsample, on=['ds', 'unique_id'], how='right')

    sc = get_all_config_results(nf)
    scores_folds.append(sc)

scores_folds[0]
flattened = [item for sublist in scores_folds for item in sublist]
df2 = pd.DataFrame(flattened)
print(df2)

# inference on estimation_train
fcst = nf.predict(df=estimation_train)

if retrain:
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_rt = NeuralForecast(models=optim_models, freq=freq_str)
    nf_rt.fit(df=estimation_train, val_size=horizon)
    fcst_rt = nf_rt.predict(df=estimation_train)

cv = fcst.merge(estimation_test, on=['ds', 'unique_id'], how='right')

# cv.to_csv(f'assets/results/{data_name},{group},dry-run.csv', index=True)

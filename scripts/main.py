import copy
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig
from src.config import N_SAMPLES, SEED
from src.cv import CV_METHODS, CV_METHODS_PARAMS
from src.cv.tw_holdout import time_wise_holdout

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
DRY_RUN = True
GROUP_IDX = 0
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

n_uids, n_trials = (30, 2) if DRY_RUN else (None, N_SAMPLES)

df, h, _, freq_str, _ = data_loader.load_everything(group, sample_n_uid=n_uids)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
est_train, est_test = data_loader.time_wise_split(df, horizon=h)


def run_cross_validation(estimation_train: pd.DataFrame,
                         estimation_test: pd.DataFrame,
                         cv_method: str,
                         nf_models: List,
                         horizon: int,
                         freq: str,
                         random_state: int):
    cv = CV_METHODS[cv_method](**CV_METHODS_PARAMS[cv_method])

    uids = estimation_train['unique_id'].unique()

    cv_results, cv_folds_config_scores = [], []
    for j, (train_index, test_index) in enumerate(cv.split(uids)):
        print(f"Fold {j}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        models_ = copy.deepcopy(nf_models)
        nf = NeuralForecast(models=models_, freq=freq)

        fold_train, fold_test = cv.get_sets_from_idx(df=estimation_train,
                                                     uids=uids,
                                                     train_index=train_index,
                                                     test_index=test_index)

        f_train_in, _ = cv.time_wise_split(fold_train, horizon=horizon)
        f_test_in, f_test_out = cv.time_wise_split(fold_test, horizon=horizon)

        np.random.seed(random_state)
        nf.fit(df=f_train_in, val_size=horizon)
        fcst_outsample = nf.predict(df=f_test_in)

        fold_cv = fcst_outsample.merge(f_test_out, on=['ds', 'unique_id'], how='right')

        config_scores = ModelsConfig.get_all_config_results(nf)
        cv_folds_config_scores.append(config_scores)

        # assuming we're aggregating by series, not by fold. we can test this later
        cv_results.append(fold_cv)

    cv_inner = pd.concat(cv_results)

    # inference on estimation_train
    optim_models = ModelsConfig.get_best_configs(cv_folds_config_scores)

    nf_final = NeuralForecast(models=optim_models, freq=freq_str)
    nf_final.fit(df=estimation_train, val_size=horizon)
    fcst = nf_final.predict(df=estimation_train)

    cv = fcst.merge(estimation_test, on=['ds', 'unique_id'], how='right')

    return cv, cv_inner


if __name__ == '__main__':

    models = ModelsConfig.get_auto_nf_models(horizon=h,
                                             try_mps=False,
                                             limit_epochs=True,
                                             n_samples=N_SAMPLES)

    for method_name in CV_METHODS:
        print(f"Running cross validation for method: {method_name}")
        cv_result, cv_inner_result = run_cross_validation(cv_method=method_name,
                                                          estimation_train=est_train,
                                                          estimation_test=est_test,
                                                          freq=freq_str,
                                                          horizon=h,
                                                          nf_models=models,
                                                          random_state=SEED)

        cv_result.to_csv(f'assets/results/{data_name},{group},{method_name},outer.csv', index=True)
        cv_inner_result.to_csv(f'assets/results/{data_name},{group},{method_name},inner.csv', index=True)

    tw_cv, tw_cv_inner = time_wise_holdout(train=est_train,
                                           test=est_test,
                                           freq=freq_str,
                                           horizon=h,
                                           models=models)

    tw_cv.to_csv(f'assets/results/{data_name},{group},TimeHoldout,outer.csv', index=True)
    tw_cv_inner.to_csv(f'assets/results/{data_name},{group},TimeHoldout,inner.csv', index=True)

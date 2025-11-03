import copy
import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig
from src.config import N_SAMPLES, SEED, LIMIT_EPOCHS, TRY_MPS
from src.cv import CV_METHODS, CV_METHODS_PARAMS
from src.cv.tw_holdout import time_wise_holdout
from src.neuralforecast_ext import NeuralForecast2

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
GROUP_IDX = 0
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

results_dir = Path('../assets/results')
# results_dir = Path('./assets/results')
results_dir.mkdir(parents=True, exist_ok=True)

# df, h, _, freq_str, _ = data_loader.load_everything(group, sample_n_uid=30)
df, h, _, freq_str, freq_i = data_loader.load_everything(group)

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
est_train, est_test = data_loader.time_wise_split(df, horizon=h * 3)


def run_cross_validation(estimation_train: pd.DataFrame,
                         estimation_test: pd.DataFrame,
                         complete_df: pd.DataFrame,
                         cv_method: str,
                         nf_models: List,
                         horizon: int,
                         freq: str,
                         freq_int: int,
                         random_state: int):
    cv = CV_METHODS[cv_method](**CV_METHODS_PARAMS[cv_method])

    uids = estimation_train['unique_id'].unique()

    cv_results, cv_folds_config_scores = [], []
    for j, (train_index, test_index) in enumerate(cv.split(uids)):
        print(f"Fold {j}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        train_uids, test_uids = uids[train_index], uids[test_index]

        models_ = copy.deepcopy(nf_models)
        nf = NeuralForecast2(models=models_, freq=freq, train_uids=train_uids)
        np.random.seed(random_state)
        cv_nf = nf.cross_validation(df=estimation_train,
                                    val_size=horizon,
                                    test_size=None,
                                    step_size=1,
                                    n_windows=horizon)

        sf_inner = StatsForecast(
            models=[SeasonalNaive(season_length=freq_int)],
            freq=freq_str,
            n_jobs=1)

        cv_sf = sf_inner.cross_validation(df=estimation_train,
                                          test_size=None,
                                          step_size=1,
                                          n_windows=horizon,
                                          h=horizon)

        cv = cv_nf.merge(cv_sf.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])
        cv = cv[cv['unique_id'].isin(test_uids)]

        config_scores = ModelsConfig.get_all_config_results(nf)
        cv_folds_config_scores.append(config_scores)

        # assuming we're aggregating by series, not by fold. we can test this later
        # cv['fold'] = j

        cv_results.append(cv)

    cv_inner = pd.concat(cv_results)

    # inference on estimation_train
    print(cv_folds_config_scores)
    optim_models = ModelsConfig.get_best_configs(cv_folds_config_scores)

    nf_final = NeuralForecast(models=optim_models, freq=freq_str)
    cv_nf_f = nf_final.cross_validation(df=complete_df,
                                        val_size=horizon,
                                        test_size=horizon * 3,
                                        step_size=1,
                                        n_windows=None)

    # cv = fcst.merge(estimation_test, on=['ds', 'unique_id'], how='right')

    sf = StatsForecast(
        models=[SeasonalNaive(season_length=freq_int)],
        freq=freq_str,
        n_jobs=1)

    cv_sf_f = sf.cross_validation(df=complete_df,
                                  h=horizon,
                                  test_size=horizon * 3,
                                  step_size=1,
                                  n_windows=None)

    cv_outer = cv_nf_f.merge(cv_sf_f.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])

    return cv_outer, cv_inner


# are the naming of uids ok for bootstrap? because of repeats

if __name__ == '__main__':

    models = ModelsConfig.get_auto_nf_models(horizon=h,
                                             try_mps=TRY_MPS,
                                             limit_epochs=LIMIT_EPOCHS,
                                             n_samples=N_SAMPLES)

    print(f"Running cross validation for method: Time-wise Holdout")
    tw_cv, tw_cv_inner = time_wise_holdout(train=est_train,
                                           test=est_test,
                                           complete_df=df,
                                           freq=freq_str,
                                           freq_int=freq_i,
                                           horizon=h,
                                           models=models)

    tw_cv.to_csv(results_dir / f'{data_name},{group},TimeHoldout,outer.csv', index=False)
    tw_cv_inner.to_csv(results_dir / f'{data_name},{group},TimeHoldout,inner.csv', index=False)

    for method_name in CV_METHODS:
        print(f"Running cross validation for method: {method_name}")
        cv_result, cv_inner_result = run_cross_validation(cv_method=method_name,
                                                          estimation_train=est_train,
                                                          estimation_test=est_test,
                                                          complete_df=df,
                                                          freq=freq_str,
                                                          freq_int=freq_i,
                                                          horizon=h,
                                                          nf_models=models,
                                                          random_state=SEED)

        cv_result.to_csv(results_dir / f'{data_name},{group},{method_name},outer.csv', index=False)
        cv_inner_result.to_csv(results_dir / f'{data_name},{group},{method_name},inner.csv', index=False)

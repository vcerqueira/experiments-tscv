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

from src.neuralnets import ModelsConfig
from src.config import N_SAMPLES, SEED, LIMIT_EPOCHS, TRY_MPS
from src.cv import CV_METHODS, CV_METHODS_PARAMS
from src.cv.tw_holdout import time_wise_holdout
from src.neuralforecast_ext import NeuralForecast2
from src.chronos_data import ChronosDataset

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
target = 'monash_tourism_monthly'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)

results_dir = Path('../assets/results')
# results_dir = Path('./assets/results')
results_dir.mkdir(parents=True, exist_ok=True)

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
est_train, est_test = ChronosDataset.time_wise_split(df, horizon)


def run_cross_validation(estimation_train: pd.DataFrame,
                         estimation_test: pd.DataFrame,
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
                                    # test_size=None,
                                    test_size=horizon,
                                    step_size=1,
                                    # n_windows=horizon,
                                    n_windows=None,
                                    )

        sf_inner = StatsForecast(
            models=[SeasonalNaive(season_length=freq_int)],
            freq=freq,
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
        cv['fold'] = j

        cv_results.append(cv)

    cv_inner = pd.concat(cv_results)

    # inference on estimation_train
    optim_models = ModelsConfig.get_best_configs(cv_folds_config_scores)

    complete_df = ChronosDataset.concat_time_wise_tr_ts(estimation_train, estimation_test)

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


if __name__ == '__main__':

    models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                             try_mps=TRY_MPS,
                                             limit_epochs=LIMIT_EPOCHS,
                                             n_samples=N_SAMPLES)

    print(f"Running cross validation for method: Time-wise Holdout")
    tw_cv, tw_cv_inner = time_wise_holdout(train=est_train,
                                           test=est_test,
                                           freq=freq,
                                           freq_int=seas_len,
                                           horizon=horizon,
                                           models=models)

    tw_cv.to_csv(results_dir / f'{target},TimeHoldout,outer.csv', index=False)
    tw_cv_inner.to_csv(results_dir / f'{target},TimeHoldout,inner.csv', index=False)

    for method_name in CV_METHODS:
        print(f"Running cross validation for method: {method_name}")
        cv_result, cv_inner_result = run_cross_validation(cv_method=method_name,
                                                          estimation_train=est_train,
                                                          estimation_test=est_test,
                                                          freq=freq,
                                                          freq_int=seas_len,
                                                          horizon=horizon,
                                                          nf_models=models,
                                                          random_state=SEED)

        cv_result.to_csv(results_dir / f'{target},{method_name},outer.csv', index=False)
        cv_inner_result.to_csv(results_dir / f'{target},{method_name},inner.csv', index=False)

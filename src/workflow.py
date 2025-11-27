import copy
from typing import List, Tuple

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.neuralnets import ModelsConfig
from src.cv import CV_METHODS, CV_METHODS_PARAMS
from src.neuralforecast_ext import NeuralForecast2
from src.chronos_data import ChronosDataset
from src.config import STEP_SIZE


def run_cross_validation(in_set: pd.DataFrame,
                         out_set: pd.DataFrame,
                         cv_method: str,
                         nf_models: List,
                         horizon: int,
                         freq: str,
                         freq_int: int,
                         random_state: int,
                         out_set_multiplier: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cv = CV_METHODS[cv_method](**CV_METHODS_PARAMS[cv_method])

    uids = in_set['unique_id'].unique()

    cv_results, cv_folds_config_scores = [], []
    for j, (train_index, test_index) in enumerate(cv.split(uids)):
        print(f"Fold {j}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        train_uids, test_uids = uids[train_index], uids[test_index]

        nf_inner_setup = {
            'df': in_set, 'val_size': horizon,
            'test_size': horizon, 'step_size': STEP_SIZE, 'n_windows': None,
        }

        sf_inner_setup = {
            'df': in_set, 'test_size': horizon,
            'step_size': STEP_SIZE, 'n_windows': None, 'h': horizon
        }

        models_ = copy.deepcopy(nf_models)
        nf = NeuralForecast2(models=models_, freq=freq, train_uids=train_uids)
        np.random.seed(random_state)
        cv_nf = nf.cross_validation(**nf_inner_setup)

        sf_inner = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq)

        cv_sf = sf_inner.cross_validation(**sf_inner_setup)

        cv = cv_nf.merge(cv_sf.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])
        cv = cv[cv['unique_id'].isin(test_uids)]

        config_scores = ModelsConfig.get_all_config_results(nf)
        cv_folds_config_scores.append(config_scores)

        # assuming we're aggregating by series, not by fold. we can test this later
        cv['fold'] = j

        cv_results.append(cv)

    cv_inner = pd.concat(cv_results)

    cvi_grouped = cv_inner.groupby(['fold', 'unique_id']).ngroup()
    cvi_change_points = cvi_grouped != cvi_grouped.shift(1)
    cvi_groups = cvi_change_points.cumsum() - 1

    cv_inner['unique_id'] = (
        cvi_groups.pipe(lambda s: (
                cv_inner['unique_id'].astype(str)
                + '_fold' + cv_inner['fold'].astype(str)
                + '_x' + s.astype(str)
        ))
    )

    optim_models = ModelsConfig.get_best_configs(cv_folds_config_scores)

    complete_df = ChronosDataset.concat_time_wise_tr_ts(in_set, out_set)

    nf_outer_setup = {
        'df': complete_df, 'val_size': horizon,
        'test_size': horizon * out_set_multiplier,
        'step_size': STEP_SIZE, 'n_windows': None
    }

    sf_outer_setup = {
        'df': complete_df, 'h': horizon,
        'test_size': horizon * out_set_multiplier,
        'step_size': STEP_SIZE, 'n_windows': None
    }

    nf_final = NeuralForecast(models=optim_models, freq=freq)
    cv_nf_f = nf_final.cross_validation(**nf_outer_setup)

    sf = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq)

    cv_sf_f = sf.cross_validation(**sf_outer_setup)

    cv_outer = cv_nf_f.merge(cv_sf_f.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])

    return cv_outer, cv_inner

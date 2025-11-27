import copy
from typing import List, Tuple

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from src.neuralnets import ModelsConfig
from src.chronos_data import ChronosDataset
from src.config import SEED, STEP_SIZE


def time_wise_holdout(in_set: pd.DataFrame,
                      out_set: pd.DataFrame,
                      models: List,
                      freq: str,
                      freq_int: int,
                      horizon: int,
                      out_set_multiplier: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # -- setup
    models_ = copy.deepcopy(models)
    nf = NeuralForecast(models=models_, freq=freq)
    sf_inner = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq)

    # -- inner cv,
    nf_inner_setup = {
        'df': in_set, 'val_size': horizon,
        'test_size': horizon, 'step_size': STEP_SIZE, 'n_windows': None,
    }

    sf_inner_setup = {
        'df': in_set, 'test_size': horizon,
        'step_size': STEP_SIZE, 'n_windows': None, 'h': horizon
    }

    np.random.seed(SEED)
    cv_inner = nf.cross_validation(**nf_inner_setup)
    cv_inner['fold'] = 0
    cv_sf_inner = sf_inner.cross_validation(**sf_inner_setup)
    cv_inner = cv_inner.merge(cv_sf_inner.drop(columns=['y', 'cutoff']), on=['ds', 'unique_id'], how='left')

    # -- cv "inference"
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_final = NeuralForecast(models=optim_models, freq=freq)

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

    cv_nf_f = nf_final.cross_validation(**nf_outer_setup)

    sf = StatsForecast(models=[SeasonalNaive(season_length=freq_int)], freq=freq)

    cv_sf_f = sf.cross_validation(**sf_outer_setup)

    cv_outer = cv_nf_f.merge(cv_sf_f.drop(columns=['y']), on=['ds', 'unique_id', 'cutoff'])

    return cv_outer, cv_inner

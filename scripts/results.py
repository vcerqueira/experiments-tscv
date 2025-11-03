import os
from functools import partial

import numpy as np
import pandas as pd

from utilsforecast.losses import smape, mape, mae, rmae, rmse, msse
from modelradar.evaluate.radar import ModelRadar

from src.cv import CV_METHODS

RESULTS_DIR = "assets/results"
DATASET = 'M3,Monthly'

rmae_sn = partial(rmae, baseline="SeasonalNaive")
# rmae_sn = smape

cv_methods = [*CV_METHODS] + ['TimeHoldout']

cv_scores = []
for method in cv_methods:
    inner_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},inner.csv")
    outer_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},outer.csv")

    if not os.path.isfile(inner_path) or not os.path.isfile(outer_path):
        continue

    cv_inner = pd.read_csv(inner_path)
    cv_inner.rename(columns={col: col.replace('Auto', '', 1) for col in cv_inner.columns if col.startswith('Auto')},
                    inplace=True)
    cv_outer = pd.read_csv(outer_path)

    radar_outer = ModelRadar(
        cv_df=cv_outer,
        metrics=[rmae_sn],
        model_names=["KAN", 'PatchTST', 'NBEATS',
                     'TiDE', 'NLinear', "MLP",
                     'DLinear', 'NHITS', 'DeepNPTS',
                     "SeasonalNaive"],
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    err_outer = radar_outer.evaluate(keep_uids=False)

    radar_inner = ModelRadar(
        cv_df=cv_inner,
        metrics=[rmae_sn],
        model_names=["KAN", 'PatchTST', 'NBEATS',
                     'TiDE', 'NLinear', "MLP",
                     'DLinear', 'NHITS', 'DeepNPTS',
                     "SeasonalNaive"],
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    err_inner = radar_inner.evaluate(keep_uids=False)

    # cv_inner_g = cv_inner.groupby('fold')
    # folds_res = []
    # for g, fold_cv in cv_inner_g:
    #
    #     radar_inner_ = ModelRadar(
    #         cv_df=fold_cv,
    #         metrics=[rmae_sn],
    #         model_names=["KAN", "MLP", 'DLinear', 'NHITS', 'DeepNPTS', "SeasonalNaive"],
    #         hardness_reference="SeasonalNaive",
    #         ratios_reference="SeasonalNaive",
    #     )
    #
    #     g_err_inner = radar_inner_.evaluate(keep_uids=False)
    #     folds_res.append(g_err_inner)
    #
    # err_inner = pd.DataFrame(folds_res).mean()

    selected_model = err_inner.idxmin()
    best_model = err_outer.idxmin()

    mae_all = (err_inner - err_outer).abs().mean()
    me_all = (err_inner - err_outer).mean()
    # mean_sq_err = ((err_inner - err_outer) ** 2).mean()
    accuracy = int(selected_model == best_model)
    regret = err_outer[selected_model] - err_outer[best_model]
    mae_best = err_outer[best_model] - err_inner[best_model]
    mae_sele = err_outer[selected_model] - err_inner[selected_model]

    cv_scores.append(
        {
            'method': method,
            'mae_all': mae_all,
            'me_all': me_all,
            'mae_best': mae_best,
            'mae_sele': mae_sele,
            'accuracy': accuracy,
            'regret': regret,
        }
    )

pd.DataFrame(cv_scores)

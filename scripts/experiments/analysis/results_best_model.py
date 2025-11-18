import os
from functools import partial

import numpy as np
import pandas as pd

from utilsforecast.losses import smape, mape, mae, rmae, rmse, msse
from modelradar.evaluate.radar import ModelRadar

from src.cv import CV_METHODS

RESULTS_DIR = "assets/results"
DATASET = 'monash_tourism_monthly'
MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]

rmae_sn = partial(rmae, baseline="SeasonalNaive")
# rmae_sn = smape
# rmae_sn = rmse

cv_methods = [*CV_METHODS] + ['TimeHoldout']

# todo can do the analysis model by model

cv_scores = []
for method in cv_methods:
    # if method in ["KFold",'RepeatedKFold']:
    #     continue
    outer_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},outer.csv")

    cv_outer = pd.read_csv(outer_path)

    radar_outer = ModelRadar(
        cv_df=cv_outer,
        metrics=[rmae_sn],
        model_names=MODELS,
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    err_outer = radar_outer.evaluate(keep_uids=False)

    best_model = err_outer.idxmin()

    scr = {
        'method': method,
        'best_model': best_model,
        **err_outer
    }

    cv_scores.append(scr)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(pd.DataFrame(cv_scores).round(3))

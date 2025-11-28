import os

import pandas as pd
from modelradar.evaluate.radar import ModelRadar

from utilsforecast.losses import mae
from src.chronos_data import ChronosDataset
from src.mase import mase_scaling_factor
from src.config import OUT_SET_MULTIPLIER
from src.cv import CV_METHODS

RESULTS_DIR = "assets/results"
DATASET = 'monash_m3_monthly'
MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]

df, horizon, _, _, seas_len = ChronosDataset.load_everything(DATASET)
in_set, _ = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)
mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=in_set)

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
        metrics=[mae],
        model_names=MODELS,
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    err_outer_uids = radar_outer.evaluate(keep_uids=True)
    err_outer = err_outer_uids.div(mase_sf, axis=0).mean()
    err_outer = err_outer.drop('SeasonalNaive')

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

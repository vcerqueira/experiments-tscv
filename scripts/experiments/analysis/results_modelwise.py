"""
This one is more exploratory - meaning, independent of selection
"""
import os

import pandas as pd
from modelradar.evaluate.radar import ModelRadar

from utilsforecast.losses import mae
from src.chronos_data import ChronosDataset
from src.mase import mase_scaling_factor
from src.config import OUT_SET_MULTIPLIER
from src.cv import CV_METHODS

RESULTS_DIR = "assets/results2"
dataset_names = set(f.split(',')[0] for f in os.listdir(RESULTS_DIR))

MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]

cv_scores = []
for ds in dataset_names:
    print(ds)

    df, horizon, _, _, seas_len = ChronosDataset.load_everything(ds)
    in_set, _ = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)
    mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=in_set)

    cv_methods = [*CV_METHODS] + ['TimeHoldout']

    for method in cv_methods:
        outer_path = os.path.join(RESULTS_DIR, f"{ds},{method},outer.csv")

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

        scr = {'method': method, 'dataset': ds, **err_outer}

        cv_scores.append(scr)

cv_df = pd.DataFrame(cv_scores).set_index(['method', 'dataset'])

ranked_df1 = cv_df.groupby('dataset').rank(axis=0, method='min')
avg_rank_per_method = ranked_df1.groupby('method').mean().mean(axis=1)
print(avg_rank_per_method)

ranked_df2 = cv_df.groupby('dataset').rank(axis=1, method='min')
avg_rank_per_nn = ranked_df2.groupby('method').mean().mean()
print(avg_rank_per_nn)

import os

import pandas as pd
import numpy as np

from utilsforecast.losses import mae
from modelradar.evaluate.radar import ModelRadar
from src.chronos_data import ChronosDataset

from src.cv import CV_METHODS
from src.mase import mase_scaling_factor
from src.utils import rename_uids
from src.config import OUT_SET_MULTIPLIER

RESULTS_DIR = "assets/results"
DATASET = 'monash_m3_monthly'
MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]

df, horizon, _, _, seas_len = ChronosDataset.load_everything(DATASET)
est_train, _ = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)

mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=est_train)

cv_methods = [*CV_METHODS] + ['TimeHoldout']

cv_scores = []
for method in cv_methods:
    # method = 'Holdout'

    inner_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},inner.csv")
    outer_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},outer.csv")

    if not os.path.isfile(inner_path) or not os.path.isfile(outer_path):
        continue

    cv_inner = pd.read_csv(inner_path)
    cv_inner.rename(columns={col: col.replace('Auto', '', 1)
                             for col in cv_inner.columns if col.startswith('Auto')},
                    inplace=True)
    cv_outer = pd.read_csv(outer_path)

    radar_outer = ModelRadar(
        cv_df=cv_outer,
        metrics=[mae],
        model_names=MODELS,
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    # err_outer = radar_outer.evaluate(keep_uids=False)
    # err_outer /= mase_sf.mean()
    err_outer_uids = radar_outer.evaluate(keep_uids=True)
    err_outer = err_outer_uids.div(mase_sf, axis=0)  # .mean()
    err_outer = err_outer.drop(columns=['SeasonalNaive'])

    radar_inner = ModelRadar(
        cv_df=cv_inner,
        metrics=[mae],
        model_names=MODELS,
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    # err_inner = radar_inner.evaluate(keep_uids=False)
    # err_inner /= mase_sf.mean()
    err_inner_uids = radar_inner.evaluate(keep_uids=True)
    err_inner_uids = rename_uids(err_inner_uids)
    err_inner = err_inner_uids.div(mase_sf.loc[err_inner_uids.index], axis=0)# .mean()
    err_inner = err_inner.drop(columns=['SeasonalNaive'])

    err_outer = err_outer.loc[err_inner.index]

    scores_list = []
    for idx, row in err_outer.iterrows():
        # idx
        # inner_row = err_inner.loc[idx]
        try:
            inner_row = err_inner.loc[idx]
            if len(inner_row.shape) > 1:
                inner_row = inner_row.mean()
        except KeyError:
            continue

        selected_model = inner_row.idxmin()
        best_model = row.idxmin()

        mae_all = (inner_row - row).abs().mean()
        me_all = (inner_row - row).mean()
        perc_under = ((err_inner - err_outer) < 0).mean()
        accuracy = int(selected_model == best_model)
        regret = row[selected_model] - row[best_model]
        mae_best = np.abs(row[best_model] - inner_row[best_model]).mean()
        mae_sele = np.abs(row[selected_model] - inner_row[selected_model]).mean()

        scores_list.append({
            'method': method,
            'uid': idx,
            'mae_all': mae_all,
            'me_all': me_all,
            'perc_under': perc_under,
            'mae_best': mae_best,
            'mae_sele': mae_sele,
            'accuracy': accuracy,
            'regret': regret,
        })

    sc = pd.DataFrame(scores_list).mean(numeric_only=True)
    sc['method'] = method

    cv_scores.append(sc)

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
print(pd.DataFrame(cv_scores).round(3))

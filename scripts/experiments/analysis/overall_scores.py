import os

import pandas as pd

from utilsforecast.losses import mae
from modelradar.evaluate.radar import ModelRadar
from src.chronos_data import ChronosDataset

from src.cv import CV_METHODS
from src.mase import mase_scaling_factor
from src.utils import rename_uids

RESULTS_DIR = "assets/results"
DATASET = 'monash_m3_monthly'
MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]
FOLD_BASED_ERROR = False

df, horizon, _, _, seas_len = ChronosDataset.load_everything(DATASET)
est_train, _ = ChronosDataset.time_wise_split(df, horizon * 4)

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
    err_outer = err_outer_uids.div(mase_sf, axis=0).fillna(0).mean()
    err_outer = err_outer.drop('SeasonalNaive')

    if FOLD_BASED_ERROR:
        cv_inner_g = cv_inner.groupby('fold')
        folds_res = []
        for g, fold_cv in cv_inner_g:
            fold_radar_inner = ModelRadar(
                cv_df=fold_cv,
                metrics=[mae],
                model_names=MODELS,
                hardness_reference="SeasonalNaive",
                ratios_reference="SeasonalNaive",
            )

            f_err_inner_uids = fold_radar_inner.evaluate(keep_uids=True)
            f_err_inner_uids = rename_uids(f_err_inner_uids)
            f_err_inner = f_err_inner_uids.div(mase_sf, axis=0).mean()
            f_err_inner = f_err_inner.drop('SeasonalNaive')
            folds_res.append(f_err_inner)

        err_inner = pd.DataFrame(folds_res).mean()
    else:
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
        err_inner = err_inner_uids.div(mase_sf, axis=0).mean()
        err_inner = err_inner.drop('SeasonalNaive')

    selected_model = err_inner.idxmin()
    best_model = err_outer.idxmin()

    mae_all = (err_inner - err_outer).abs().mean()
    me_all = (err_inner - err_outer).mean()
    perc_under = ((err_inner - err_outer) < 0).mean()
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
            'perc_under': perc_under,
            'mae_best': mae_best,
            'mae_sele': mae_sele,
            'accuracy': accuracy,
            'regret': regret,
        }
    )

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
print(pd.DataFrame(cv_scores).round(3))

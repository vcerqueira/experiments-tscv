import os

import pandas as pd
from modelradar.evaluate.radar import ModelRadar

from utilsforecast.losses import mae
from src.chronos_data import ChronosDataset
from src.mase import mase_scaling_factor
from src.config import OUT_SET_MULTIPLIER
from src.cv import CV_METHODS
from src.utils import rename_uids

RESULTS_DIR = "assets/results"

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

    FOLD_BASED_ERROR = False
    cv_methods = [*CV_METHODS] + ['TimeHoldout']

    for method in cv_methods:
        # if method in ["KFold",'RepeatedKFold']:
        #     continue
        inner_path = os.path.join(RESULTS_DIR, f"{ds},{method},inner.csv")
        outer_path = os.path.join(RESULTS_DIR, f"{ds},{method},outer.csv")

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
        err_outer = err_outer_uids.div(mase_sf, axis=0).mean()
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
            err_inner = err_inner_uids.div(mase_sf.loc[err_inner_uids.index], axis=0).mean()
            err_inner = err_inner.drop('SeasonalNaive')

        selected_model = err_inner.idxmin()
        best_model = err_outer.idxmin()

        scr = {
            'Method': method,
            'Dataset': ds,
            'selected_error': err_outer[selected_model],
            'best_error': err_outer[best_model],
        }

        cv_scores.append(scr)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

cv_df = pd.DataFrame(cv_scores).set_index('Method')

print(cv_df.round(3))

cv_df['outer_regret'] = cv_df['selected_error'] - cv_df['best_error']

cv_pivot = cv_df.reset_index().pivot(index='Dataset', columns='Method', values='selected_error')

cv_pivot_ext = cv_pivot.copy()
cv_pivot_ext.loc['Avg. Rank'] = cv_pivot.rank(axis=1).mean()
cv_pivot_ext.loc['Avg'] = cv_pivot.mean()

print(cv_pivot_ext.round(3))

print(cv_df.groupby('Method').mean(numeric_only=True))

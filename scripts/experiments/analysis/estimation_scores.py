import os

import pandas as pd

from utilsforecast.losses import mae
from modelradar.evaluate.radar import ModelRadar
from src.loaders import ChronosDataset, LongHorizonDatasetR

from src.cv import CV_METHODS
from src.mase import mase_scaling_factor
from src.utils import (rename_uids,
                       to_latex_tab,
                       METHOD_NAME_MAPPING)
from src.config import OUT_SET_MULTIPLIER, FOLD_BASED_ERROR

RESULTS_DIR = "assets/results"

dataset_names = set(f.split(',')[0] for f in os.listdir(RESULTS_DIR))

MODELS = ["KAN", 'PatchTST', 'NBEATS', 'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear', 'NHITS', 'DeepNPTS',
          "SeasonalNaive"]

cv_scores = []
all_cv_scores = []
for ds in dataset_names:
    print(ds)

    if ds in [*LongHorizonDatasetR.FREQUENCY_MAP]:
        df, horizon, _, _, seas_len = LongHorizonDatasetR.load_everything(ds)
    else:
        df, horizon, _, _, seas_len = ChronosDataset.load_everything(ds)

    if ds == 'Weather':
        seas_len = 30

    in_set, _ = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)
    dev_set, _ = ChronosDataset.time_wise_split(in_set, horizon)
    mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=in_set)
    inner_mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=dev_set)

    cv_methods = [*CV_METHODS] + ['TimeHoldout']

    for method in cv_methods:
        print(method)
        inner_path = os.path.join(RESULTS_DIR, f"{ds},{method},inner.csv")
        outer_path = os.path.join(RESULTS_DIR, f"{ds},{method},outer.csv")

        if not os.path.isfile(inner_path) or not os.path.isfile(outer_path):
            continue

        cv_inner = pd.read_csv(inner_path)
        cv_inner.rename(columns={col: col.replace('Auto', '', 1)
                                 for col in cv_inner.columns if col.startswith('Auto')},
                        inplace=True)
        cv_outer = pd.read_csv(outer_path)

        # cv_outer.rename(columns={col: col.replace('Auto', '', 1)
        #                          for col in cv_outer.columns if col.startswith('Auto')},
        #                 inplace=True)

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
                # f_err_inner = f_err_inner_uids.div(mase_sf, axis=0).mean()
                f_err_inner = f_err_inner_uids.div(inner_mase_sf, axis=0).mean()
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

            err_inner_uids = radar_inner.evaluate(keep_uids=True)
            err_inner_uids = rename_uids(err_inner_uids)
            # err_inner = err_inner_uids.div(mase_sf.loc[err_inner_uids.index], axis=0).mean()
            err_inner = err_inner_uids.div(inner_mase_sf.loc[err_inner_uids.index], axis=0).mean()
            err_inner = err_inner.drop('SeasonalNaive')

        selected_model = err_inner.idxmin()
        best_model = err_outer.idxmin()

        mae_all = (err_inner - err_outer).abs()
        mae_avg = mae_all.mean()
        me_all = (err_inner - err_outer)
        perc_under = ((err_inner - err_outer) < 0).mean()
        accuracy = int(selected_model == best_model)
        regret = err_outer[selected_model] - err_outer[best_model]
        mae_best = err_outer[best_model] - err_inner[best_model]
        mae_sele = err_outer[selected_model] - err_inner[selected_model]

        cv_scores.append(
            {
                'Dataset': ds,
                'Method': method,
                'MAPEE': mae_avg,
                'Accuracy': accuracy,
                'Regret': regret,
            }
        )

        for m in MODELS[:-1]:
            all_cv_scores.append(
                {
                    'Dataset': ds,
                    'Method': method,
                    'Model' : m,
                    'MAPEE': mae_all[m],
                    'MPEE' : me_all[m],
                }
            )

# pd.set_option('display.max_columns', 10)
# pd.set_option('display.max_rows', 30)

cv_df = pd.DataFrame(cv_scores)
all_cv_df = pd.DataFrame(all_cv_scores)
cv_df.groupby('Method').mean(numeric_only=True)

cv_df_summ = cv_df.groupby('Method').mean(numeric_only=True).round(3)
cv_df_summ = pd.merge(cv_df_summ.reset_index(), all_cv_df.groupby('Method')['MPEE'].apply(lambda x: ((x < 0).mean()).round(3)).reset_index(name='Overoptimism'))
cv_df_summ = cv_df_summ.rename(index=METHOD_NAME_MAPPING)
print(cv_df_summ)
print(to_latex_tab(cv_df_summ.T, round_to_n=3, rotate_cols=False))

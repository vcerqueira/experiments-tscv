import os
from functools import partial

import numpy as np
import pandas as pd

from utilsforecast.losses import smape, mape, mae, rmae
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

cv_methods = ['TimeHoldout'] + [*CV_METHODS]

cv_scores = []
for method in cv_methods:
    print(method)
    inner_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},inner.csv")
    outer_path = os.path.join(RESULTS_DIR, f"{DATASET},{method},outer.csv")

    if not os.path.isfile(inner_path) or not os.path.isfile(outer_path):
        continue

    cv_inner = pd.read_csv(inner_path)
    cv_inner.rename(columns={col: col.replace('Auto', '', 1) for col in cv_inner.columns if col.startswith('Auto')},
                    inplace=True)
    cv_outer = pd.read_csv(outer_path)

    radar_inner = ModelRadar(
        cv_df=cv_inner,
        metrics=[rmae_sn],
        model_names=MODELS,
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    radar_outer = ModelRadar(
        cv_df=cv_outer,
        metrics=[rmae_sn],
        model_names=MODELS,
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )

    # todo se aggregar por uid, tenho de ter em conta repeticoes
    # ... basta iterar pelo err_inner, não? não, porque aqui já faltam dados. ja houve agg por uid
    err_inner = radar_inner.evaluate(keep_uids=True)


    # err_outer = radar_outer.evaluate(keep_uids=True)

    def base_uid(x):
        parts = x.split('_')
        if len(parts) > 1:
            return '_'.join(parts[:-2])
        else:
            return x


    if "fold" in err_inner.index[0]:
        base_uid_list = err_inner.index.str.split('_').map(
            lambda x: '_'.join(x[:-2]) if len(x) > 2 else err_inner.index[0])
        err_inner_cln = err_inner.copy()
        err_inner_cln.index = base_uid_list
    else:
        err_inner_cln = err_inner.copy()

    err_outer = radar_outer.evaluate(keep_uids=True).loc[err_inner_cln.index]
    err_outer = err_outer.groupby('unique_id').mean()

    scores_list = []
    for idx, row in err_outer.iterrows():
        # inner_row = err_inner.loc[idx]
        try:
            inner_row = err_inner_cln.loc[idx]
            if len(inner_row.shape) > 1:
                inner_row = inner_row.mean()
        except KeyError:
            continue

        selected_model = inner_row.idxmin()
        best_model = row.idxmin()

        mae_all = (inner_row - row).abs().mean()
        me_all = (inner_row - row).mean()
        mean_sq_err = ((inner_row - row) ** 2).mean()
        accuracy = int(selected_model == best_model)
        regret = row[selected_model] - row[best_model]
        mae_best = np.abs(row[best_model] - inner_row[best_model]).mean()
        mae_sele = np.abs(row[selected_model] - inner_row[selected_model]).mean()

        scores_list.append({
            'method': method,
            'uid': idx,
            'mae_all': mae_all,
            'me_all': me_all,
            'mae_best': mae_best,
            'mae_sele': mae_sele,
            'accuracy': accuracy,
            'regret': regret,
        })

    sc = pd.DataFrame(scores_list).mean(numeric_only=True)
    sc['method'] = method

    cv_scores.append(sc)

print(pd.DataFrame(cv_scores).set_index('method').round(3))

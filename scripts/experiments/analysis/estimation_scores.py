import os
from pathlib import Path

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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

RESULTS_BASE = Path("assets/results_holdout")

MODELS = ["KAN",
          # 'PatchTST',
          'NBEATS',
          # 'TFT',
          'TiDE',
          'NLinear',
          "MLP",
          'DLinear',
          # 'NHITS',
          'DeepNPTS',
          "SeasonalNaive"]


def discover_seed_dirs(base: Path) -> list[tuple[str, Path]]:
    if not base.is_dir():
        return []
    seed_dirs = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and child.name.startswith("seed_"):
            seed_dirs.append((child.name, child))
    return seed_dirs


def dataset_names_in_dir(results_dir: Path) -> set[str]:
    return {f.name.split(',')[0] for f in results_dir.iterdir() if f.suffix == '.csv'}


def scores_for_dataset_method(
        ds: str,
        method: str,
        results_dir: Path,
) -> dict | None:
    inner_path = results_dir / f"{ds},{method},inner.csv"
    outer_path = results_dir / f"{ds},{method},outer.csv"

    if not inner_path.is_file() or not outer_path.is_file():
        return None

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

    err_outer_uids = radar_outer.evaluate(keep_uids=True)
    err_outer = err_outer_uids.div(mase_sf, axis=0).mean()
    err_outer = err_outer.drop('SeasonalNaive')

    if FOLD_BASED_ERROR:
        cv_inner_g = cv_inner.groupby('fold')
        folds_res = []
        for _, fold_cv in cv_inner_g:
            fold_radar_inner = ModelRadar(
                cv_df=fold_cv,
                metrics=[mae],
                model_names=MODELS,
                hardness_reference="SeasonalNaive",
                ratios_reference="SeasonalNaive",
            )

            f_err_inner_uids = fold_radar_inner.evaluate(keep_uids=True)
            f_err_inner_uids = rename_uids(f_err_inner_uids)
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
        err_inner = err_inner_uids.div(inner_mase_sf.loc[err_inner_uids.index], axis=0).mean()
        err_inner = err_inner.drop('SeasonalNaive')

    selected_model = err_inner.idxmin()
    best_model = err_outer.idxmin()

    mae_all = (err_inner - err_outer).abs().mean()
    me_all = (err_inner - err_outer).mean()
    accuracy = int(selected_model == best_model)
    regret = err_outer[selected_model] - err_outer[best_model]

    return {
        'Dataset': ds,
        'Method': method,
        'MAPEE': mae_all,
        'MPEE': me_all,
        'Accuracy': accuracy,
        'Regret': regret,
    }


def compute_scores_for_seed(seed_label: str, results_dir: Path) -> pd.DataFrame:
    cv_methods = [*CV_METHODS] + ['TimeHoldout']
    dataset_names = dataset_names_in_dir(results_dir)

    rows = []
    for ds in sorted(dataset_names):
        print(f"{seed_label} / {ds}")
        for method in cv_methods:
            print(f"  {method}")
            row = scores_for_dataset_method(ds, method, results_dir)
            if row is not None:
                row['Seed'] = seed_label
                rows.append(row)

    return pd.DataFrame(rows)


def summarize(cv_df: pd.DataFrame) -> pd.DataFrame:
    cv_df_summ = cv_df.groupby('Method').mean(numeric_only=True).round(3)
    cv_df_summ["Perc. underestimates"] = (
        cv_df.groupby('Method')['MPEE'].apply(lambda x: round(100 * (x < 0).mean(), 2))
    )
    cv_df_summ["Avg. under-estimate"] = (
        cv_df.groupby('Method')['MPEE'].apply(
            lambda x: round(x[x < 0].mean(), 3) if (x < 0).any() else float('nan')
        )
    )
    cv_df_summ["Avg. over-estimate"] = (
        cv_df.groupby('Method')['MPEE'].apply(
            lambda x: round(x[x > 0].mean(), 3) if (x > 0).any() else float('nan')
        )
    )
    return cv_df_summ.rename(index=METHOD_NAME_MAPPING)


seed_dirs = discover_seed_dirs(RESULTS_BASE)

per_seed_dfs = []
for seed_label, seed_path in seed_dirs:
    print(f"=== {seed_label} ===")
    per_seed_dfs.append(compute_scores_for_seed(seed_label, seed_path))

cv_df_by_seed = pd.concat(per_seed_dfs, ignore_index=True)

# Average metrics across seeds for each (dataset, method)
cv_df = (
    cv_df_by_seed
    .groupby(['Dataset', 'Method'], as_index=False)
    .mean(numeric_only=True)
)

cv_df_summ = summarize(cv_df)
print(cv_df_summ.drop("MPEE", axis=1))
print(to_latex_tab(cv_df_summ.drop("MPEE", axis=1).T, round_to_n=3, rotate_cols=False))

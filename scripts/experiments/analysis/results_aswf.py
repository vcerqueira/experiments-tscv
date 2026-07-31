import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from modelradar.evaluate.radar import ModelRadar

from utilsforecast.losses import mae
from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.mase import mase_scaling_factor
from src.config import OUT_SET_MULTIPLIER
from src.cv import CV_METHODS
from src.utils import (rename_uids,
                       to_latex_tab,
                       METHOD_NAME_MAPPING,
                       DATA_NAME_MAPPING)

RESULTS_BASE = Path("assets/results")
FOLD_BASED_ERROR = False

MODELS = ["KAN",
          'PatchTST',
          'NBEATS',
          'TFT',
          'TiDE', 'NLinear', "MLP",
          'DLinear',
          'NHITS',
          'DeepNPTS',
          "SeasonalNaive"]


def discover_seed_dirs(base: Path) -> list[tuple[str, Path]]:
    if not base.is_dir():
        return []
    return [
        (child.name, child)
        for child in sorted(base.iterdir())
        if child.is_dir() and child.name.startswith("seed_")
    ]


def dataset_names_in_dir(results_dir: Path) -> set[str]:
    return {f.name.split(',')[0] for f in results_dir.iterdir() if f.suffix == '.csv'}


def score_dataset_method(ds: str, method: str, results_dir: Path) -> dict | None:
    inner_path = results_dir / f"{ds},{method},inner.csv"
    outer_path = results_dir / f"{ds},{method},outer.csv"

    if not inner_path.is_file() or not outer_path.is_file():
        return None

    if ds in [*LongHorizonDatasetR.FREQUENCY_MAP]:
        df, horizon, _, _, seas_len = LongHorizonDatasetR.load_everything(ds)
    else:
        df, horizon, _, _, seas_len = ChronosDataset.load_everything(ds)

    if ds in ['Weather']:
        seas_len = 1

    in_set, _ = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)
    mase_sf = mase_scaling_factor(seasonality=seas_len, train_df=in_set)

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

        err_inner_uids = radar_inner.evaluate(keep_uids=True)
        err_inner_uids = rename_uids(err_inner_uids)
        err_inner = err_inner_uids.div(mase_sf.loc[err_inner_uids.index], axis=0).mean()
        err_inner = err_inner.drop('SeasonalNaive')

    selected_model = err_inner.idxmin()
    best_model = err_outer.idxmin()

    return {
        'Method': method,
        'Dataset': ds,
        'selected_error': err_outer[selected_model],
        'best_error': err_outer[best_model],
    }


def compute_scores_for_seed(seed_label: str, results_dir: Path) -> pd.DataFrame:
    cv_methods = [*CV_METHODS] + ['TimeHoldout']
    rows = []
    for ds in sorted(dataset_names_in_dir(results_dir)):
        print(f"{seed_label} / {ds}")
        # if ds == 'monash_m1_quarterly':
        #     continue

        for method in cv_methods:
            row = score_dataset_method(ds, method, results_dir)
            if row is not None:
                row['Seed'] = seed_label
                rows.append(row)
    return pd.DataFrame(rows)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

seed_dirs = discover_seed_dirs(RESULTS_BASE)
if not seed_dirs:
    raise SystemExit(
        f"No seed_* folders found under {RESULTS_BASE.resolve()}. "
        "Expected e.g. assets/results/seed_123/"
    )

per_seed_dfs = []
for seed_label, seed_path in seed_dirs:
    print(f"=== {seed_label} ===")
    per_seed_dfs.append(compute_scores_for_seed(seed_label, seed_path))

cv_df_by_seed = pd.concat(per_seed_dfs, ignore_index=True)

cv_df = (
    cv_df_by_seed
    .groupby(['Dataset', 'Method'], as_index=False)
    .mean(numeric_only=True)
)

print(cv_df.round(3))

cv_df = cv_df.copy()
cv_df['outer_regret'] = cv_df['selected_error'] - cv_df['best_error']

cv_pivot = cv_df.pivot(index='Dataset', columns='Method', values='selected_error')

cv_pivot_ext = cv_pivot.copy()
cv_pivot_ext.loc['Avg. Rank'] = cv_pivot.rank(axis=1).mean()
cv_pivot_ext.loc['Avg'] = cv_pivot.mean()
cv_pivot_ext.loc['Top 2 Count'] = (cv_pivot.rank(axis=1, method='min') < 3).sum().astype(int)

cv_pivot_ext = cv_pivot_ext.rename(columns=METHOD_NAME_MAPPING, index=DATA_NAME_MAPPING)
cv_pivot_ext.columns.name = 'Methods'
cv_pivot_ext.index.name = 'Dataset'

print(cv_pivot_ext.round(3))
print(to_latex_tab(cv_pivot_ext, round_to_n=3, rotate_cols=False))
#
# avg_rank = cv_pivot_ext.loc['Avg. Rank'].sort_values()
# fig, ax = plt.subplots(figsize=(9, 4.5))
# avg_rank.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
# ax.set_title('Average Rank by CV Method')
# ax.set_xlabel('Methods')
# ax.set_ylabel('Avg. Rank')
# ax.grid(axis='y', linestyle='--', alpha=0.4)
# ax.set_axisbelow(True)
# plt.xticks(rotation=30, ha='right')
# plt.tight_layout()
#
# out_png = os.path.join("assets", "avg_rank_barplot.png")
# fig.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
# plt.close(fig)

cv_df.set_index('Method')['selected_error']

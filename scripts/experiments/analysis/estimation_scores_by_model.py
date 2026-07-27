from pathlib import Path

import pandas as pd

from utilsforecast.losses import mae
from modelradar.evaluate.radar import ModelRadar
from src.loaders import ChronosDataset, LongHorizonDatasetR

from src.cv import CV_METHODS
from src.mase import mase_scaling_factor
from src.utils import rename_uids, to_latex_tab
from src.config import OUT_SET_MULTIPLIER, FOLD_BASED_ERROR

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

RESULTS_BASE = Path("assets/results_backup")

MODELS = ["KAN",
          'PatchTST',
          'NBEATS',
          'TFT',
          'TiDE',
          'NLinear',
          "MLP",
          'DLinear',
          'NHITS',
          'DeepNPTS']


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


def scores_for_dataset_method(
        ds: str,
        method: str,
        results_dir: Path,
) -> pd.DataFrame | None:
    """
    Returns a DataFrame with one row per model, containing:
    - outer_error: actual performance on outer set
    - inner_error: estimated performance from inner CV
    - estimation_error: inner - outer (positive = overestimate)
    - abs_estimation_error: |inner - outer|
    """
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

    # Outer error per model
    radar_outer = ModelRadar(
        cv_df=cv_outer,
        metrics=[mae],
        model_names=MODELS + ["SeasonalNaive"],
        hardness_reference="SeasonalNaive",
        ratios_reference="SeasonalNaive",
    )
    err_outer_uids = radar_outer.evaluate(keep_uids=True)
    err_outer = err_outer_uids.div(mase_sf, axis=0).mean()

    # Inner error per model
    if FOLD_BASED_ERROR:
        cv_inner_g = cv_inner.groupby('fold')
        folds_res = []
        for _, fold_cv in cv_inner_g:
            fold_radar_inner = ModelRadar(
                cv_df=fold_cv,
                metrics=[mae],
                model_names=MODELS + ["SeasonalNaive"],
                hardness_reference="SeasonalNaive",
                ratios_reference="SeasonalNaive",
            )
            f_err_inner_uids = fold_radar_inner.evaluate(keep_uids=True)
            f_err_inner_uids = rename_uids(f_err_inner_uids)
            f_err_inner = f_err_inner_uids.div(inner_mase_sf, axis=0).mean()
            folds_res.append(f_err_inner)
        err_inner = pd.DataFrame(folds_res).mean()
    else:
        radar_inner = ModelRadar(
            cv_df=cv_inner,
            metrics=[mae],
            model_names=MODELS + ["SeasonalNaive"],
            hardness_reference="SeasonalNaive",
            ratios_reference="SeasonalNaive",
        )
        err_inner_uids = radar_inner.evaluate(keep_uids=True)
        err_inner_uids = rename_uids(err_inner_uids)
        err_inner = err_inner_uids.div(inner_mase_sf.loc[err_inner_uids.index], axis=0).mean()

    # Build per-model results
    rows = []
    for model in MODELS:
        if model not in err_outer.index or model not in err_inner.index:
            continue
        outer_err = err_outer[model]
        inner_err = err_inner[model]
        rows.append({
            'Dataset': ds,
            'Method': method,
            'Model': model,
            'outer_error': outer_err,
            'inner_error': inner_err,
            'estimation_error': inner_err - outer_err,  # positive = overestimate
            'abs_estimation_error': abs(inner_err - outer_err),
        })

    return pd.DataFrame(rows) if rows else None


def compute_scores_for_seed(seed_label: str, results_dir: Path) -> pd.DataFrame:
    cv_methods = [*CV_METHODS] + ['TimeHoldout']
    dataset_names = dataset_names_in_dir(results_dir)

    dfs = []
    for ds in sorted(dataset_names):
        print(f"{seed_label} / {ds}")
        for method in cv_methods:
            df = scores_for_dataset_method(ds, method, results_dir)
            if df is not None:
                df['Seed'] = seed_label
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def summarize_by_model(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model across all datasets and methods."""
    summary = cv_df.groupby('Model').agg({
        'outer_error': 'mean',
        'inner_error': 'mean',
        'estimation_error': 'mean',
        'abs_estimation_error': 'mean',
    }).round(3)

    summary.columns = ['Avg. Outer Error', 'Avg. Inner Error',
                       'Mean Est. Error', 'Mean Abs. Est. Error']

    # Add rankings
    summary['Outer Rank'] = summary['Avg. Outer Error'].rank().astype(int)
    summary['Estimation Difficulty Rank'] = summary['Mean Abs. Est. Error'].rank(ascending=False).astype(int)

    # Percentage of cases where performance was underestimated (inner < outer)
    underest_pct = cv_df.groupby('Model')['estimation_error'].apply(
        lambda x: round(100 * (x < 0).mean(), 1)
    )
    summary['% Underestimated'] = underest_pct

    return summary.sort_values('Avg. Outer Error')


def summarize_by_model_and_method(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot showing estimation error by model and CV method."""
    pivot = cv_df.pivot_table(
        index='Model',
        columns='Method',
        values='abs_estimation_error',
        aggfunc='mean'
    ).round(3)
    return pivot


seed_dirs = discover_seed_dirs(RESULTS_BASE)

per_seed_dfs = []
for seed_label, seed_path in seed_dirs:
    print(f"=== {seed_label} ===")
    per_seed_dfs.append(compute_scores_for_seed(seed_label, seed_path))

cv_df_by_seed = pd.concat(per_seed_dfs, ignore_index=True)

cv_df = (
    cv_df_by_seed
    .groupby(['Dataset', 'Method', 'Model'], as_index=False)
    .mean(numeric_only=True)
)


cv_df.groupby('Model').mean(numeric_only=True)['outer_error'].sort_values()
cv_df.groupby('Model').mean(numeric_only=True)['abs_estimation_error'].sort_values()
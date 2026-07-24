import os
import warnings
from pathlib import Path

from src.neuralnets_auto import ModelsConfig
from src.cv.sw_kfold import SeriesWiseKFold
from src.loaders import ChronosDataset, LongHorizonDatasetR
from src.workflow_nestedcv import run_cross_validation
from src.config import (N_SAMPLES,
                        SEED,
                        LIMIT_EPOCHS,
                        ENGINE,
                        OUT_SET_MULTIPLIER,
                        HOLDOUT_FOR_OUTSET)

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

N_FOLDS_VALUES = [2, 3, 5, 7, 10]

# ---- data loading and partitioning
target = 'monash_m3_monthly'
df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
# df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

RESULTS_PATH = '../../assets/results_ablation'

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
in_set_all, out_set = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)

if HOLDOUT_FOR_OUTSET > 0:
    in_set = ChronosDataset.sample_uids(in_set_all, 1 - HOLDOUT_FOR_OUTSET)
else:
    in_set = in_set_all.copy()

results_dir = Path(RESULTS_PATH) / f'seed_{SEED}'

if __name__ == '__main__':
    results_dir.mkdir(parents=True, exist_ok=True)
    print(results_dir.absolute())

    models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                             engine=ENGINE,
                                             limit_epochs=LIMIT_EPOCHS,
                                             n_samples=N_SAMPLES)

    for n_folds in N_FOLDS_VALUES:
        method_name = f'KFold_k{n_folds}'
        print(f"Running cross validation for method: {method_name}")

        # Create custom CV method params for this n_folds value
        cv_methods_override = {'KFold': SeriesWiseKFold}
        cv_params_override = {'KFold': {'n_splits': n_folds, 'random_state': SEED}}

        cv_result, cv_inner_result = run_cross_validation(
            cv_method='KFold',
            in_set=in_set,
            in_set_all=in_set_all,
            out_set=out_set,
            freq=freq,
            freq_int=seas_len,
            horizon=horizon,
            nf_models=models,
            random_state=SEED,
            out_set_multiplier=OUT_SET_MULTIPLIER,
            cv_methods_override=cv_methods_override,
            cv_params_override=cv_params_override,
        )

        cv_result.to_csv(results_dir / f'{target},{method_name},outer.csv', index=False)
        cv_inner_result.to_csv(results_dir / f'{target},{method_name},inner.csv', index=False)

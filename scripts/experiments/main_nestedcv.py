import os
import warnings
from pathlib import Path

from src.neuralnets_auto import ModelsConfig
from src.cv import CV_METHODS
from src.cv.tw_holdout_nested import time_wise_holdout
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

# ---- data loading and partitioning
target = 'Weather'
# df, horizon, _, freq, seas_len = ChronosDataset.load_everything(target)
df, horizon, _, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')

RESULTS_PATH = '../../assets/results{}'

# - split dataset by time
# -- estimation_train is used for inner cv and final training
# ----- the data we use to get performance estimations
# -- estimation_test is only used at the end to see how well our estimation worked
in_set_all, out_set = ChronosDataset.time_wise_split(df, horizon * OUT_SET_MULTIPLIER)

if HOLDOUT_FOR_OUTSET > 0:
    path_ = RESULTS_PATH.format('_holdout')
    in_set = ChronosDataset.sample_uids(in_set_all, 1 - HOLDOUT_FOR_OUTSET)
else:
    path_ = RESULTS_PATH.format('')
    in_set = in_set_all.copy()

results_dir = Path(path_) / f'seed_{SEED}'

if __name__ == '__main__':
    results_dir.mkdir(parents=True, exist_ok=True)
    print(results_dir.absolute())

    models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                             engine=ENGINE,
                                             limit_epochs=LIMIT_EPOCHS,
                                             n_samples=N_SAMPLES)

    tw_outer_path = results_dir / f'{target},TimeHoldout,outer.csv'
    tw_inner_path = results_dir / f'{target},TimeHoldout,inner.csv'

    if not (tw_outer_path.exists() and tw_inner_path.exists()):
        print(f"Running cross validation for method: Time-wise Holdout")
        tw_cv, tw_cv_inner = time_wise_holdout(in_set=in_set,
                                               in_set_all=in_set_all,
                                               out_set=out_set,
                                               freq=freq,
                                               freq_int=seas_len,
                                               horizon=horizon,
                                               models=models,
                                               out_set_multiplier=OUT_SET_MULTIPLIER)

        tw_cv.to_csv(tw_outer_path, index=False)
        tw_cv_inner.to_csv(tw_inner_path, index=False)
    else:
        print(f"Skipping Time-wise Holdout (results already exist)")

    for method_name in CV_METHODS:
        cv_outer_path = results_dir / f'{target},{method_name},outer.csv'
        cv_inner_path = results_dir / f'{target},{method_name},inner.csv'

        if not (cv_outer_path.exists() and cv_inner_path.exists()):
            print(f"Running cross validation for method: {method_name}")
            cv_result, cv_inner_result = run_cross_validation(cv_method=method_name,
                                                              in_set=in_set,
                                                              in_set_all=in_set_all,
                                                              out_set=out_set,
                                                              freq=freq,
                                                              freq_int=seas_len,
                                                              horizon=horizon,
                                                              nf_models=models,
                                                              random_state=SEED,
                                                              out_set_multiplier=OUT_SET_MULTIPLIER)

            cv_result.to_csv(cv_outer_path, index=False)
            cv_inner_result.to_csv(cv_inner_path, index=False)
        else:
            print(f"Skipping {method_name} (results already exist)")

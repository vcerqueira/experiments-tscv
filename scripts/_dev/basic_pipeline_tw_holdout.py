import os
import warnings

from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neural.methods import ModelsConfig
from src.config import N_SAMPLES, RETRAIN_FOR_TEST

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
DRY_RUN = True
GROUP_IDX = 0
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

n_uids, n_trials, retrain = (30, 2, False) if DRY_RUN else (None, N_SAMPLES, RETRAIN_FOR_TEST)

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=n_uids)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

estimation_train, estimation_test = data_loader.time_wise_split(df, horizon=horizon)

models = ModelsConfig.get_auto_nf_models(horizon=horizon, n_samples=n_trials)

# ---- model setup
nf = NeuralForecast(models=models, freq=freq_str)

cv = nf.cross_validation(df=estimation_train, val_size=horizon, test_size=horizon, n_windows=None)

fcst = nf.predict(df=estimation_train)

if retrain:
    optim_models = ModelsConfig.get_best_configs(nf)

    nf_rt = NeuralForecast(models=optim_models, freq=freq_str)
    nf_rt.fit(df=estimation_train, val_size=horizon)
    fcst_rt = nf_rt.predict(df=estimation_train)

print(fcst)

cv = fcst.merge(estimation_test, on=['ds', 'unique_id'], how='right')

# cv.to_csv(f'assets/results/{data_name},{group},dry-run.csv', index=True)

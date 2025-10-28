import os
import warnings

from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig
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

models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                         n_samples=n_trials,
                                         try_mps=False,
                                         limit_epochs=True)

# ---- model setup
nf = NeuralForecast(models=models, freq=freq_str)

cv = nf.cross_validation(df=estimation_train, val_size=horizon, test_size=horizon, n_windows=None)


nf.models[0].results_

from pprint import pprint
pprint(nf.models[0].results[0].metrics)
pprint(nf.models[0].results[1].metrics)
pprint(nf.models[0].results.get_best_result().metrics)

fcst = nf.predict(df=estimation_train)

optim_models = []
for mod in nf.models:
    opm_mod = ModelsConfig.MODEL_CLASSES[mod.alias](**mod.results.get_best_result().config)

    optim_models.append(opm_mod)

nf_rt = NeuralForecast(models=optim_models, freq=freq_str)
nf_rt.fit(df=estimation_train, val_size=horizon)
fcst_rt = nf_rt.predict(df=estimation_train)

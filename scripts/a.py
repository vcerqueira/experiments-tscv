import os
import warnings

import pandas as pd
from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neural.methods import ModelsConfig
from src.config import N_SAMPLES

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
GROUP_IDX = 0
EXPERIMENT = 'nf'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=30)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

estimation_train, estimation_test = data_loader.train_test_split(df, horizon=horizon)

models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                         n_samples=N_SAMPLES)

# ---- model setup
nf = NeuralForecast(models=models, freq=freq_str)

cv = nf.cross_validation(df=estimation_train,
                         val_size=12,
                         test_size=12,
                         n_windows=None)

print(cv)

# how to retrain the best configuration only?
# nf.models[0].results.get_best_result().config

nf.fit(df=estimation_train, use_init_models=False)
#

fcst = nf.predict(df=estimation_train)
#
#
# # ---- model fitting
# nf.fit(df=train)
# nf.predict()
# #
#
#
#
#
# best_conf_df.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=True)

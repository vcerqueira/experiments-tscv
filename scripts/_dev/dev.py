import os
import warnings

from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from src.neuralnets import ModelsConfig

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

# ---- data loading and partitioning
GROUP_IDX = 0
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]


df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=30)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

estimation_train, estimation_test = data_loader.time_wise_split(df, horizon=horizon*5)

models = ModelsConfig.get_auto_nf_models(horizon=horizon,
                                         n_samples=2,
                                         try_mps=False,
                                         limit_epochs=True)[:2]


# ---- model setup
nf = NeuralForecast(models=models, freq=freq_str)
nf.fit(df=estimation_train)

# Pure inference - no retraining
# For h*3 predictions with rolling origin step_size=1
predictions = nf.predict(df=estimation_test, step_size=1, h=horizon*2)
nf.cross_validation()
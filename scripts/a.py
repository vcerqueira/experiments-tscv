import warnings

import pandas as pd
from neuralforecast import NeuralForecast

from src.load_data.config import DATASETS, DATA_GROUPS
from utils.models_config import ModelsConfig

warnings.filterwarnings('ignore')

# ---- data loading and partitioning
GROUP_IDX = 7
EXPERIMENT = 'hpo-nf'
data_name, group = DATA_GROUPS[GROUP_IDX]
print(data_name, group)
data_loader = DATASETS[data_name]

df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group)
# df, horizon, n_lags, freq_str, freq_int = data_loader.load_everything(group, sample_n_uid=30)
# df = data_loader.get_uid_tails(df, tail_size=100)
# df = data_loader.dummify_series(df)

train, _ = data_loader.train_test_split(df, horizon=horizon)

# ---- model setup
nf = NeuralForecast(models=ModelsConfig.get_auto_nf_models(horizon=horizon,
                                                           limit_val_batches=True), freq=freq_str)

# ---- model fitting
nf.fit(df=train)

best_configs = {}
for mod in nf.models:
    best_configs[mod.alias] = mod.results.get_best_result().config
    # for MLF
    # best_config = nf.results_['ModelName'].best_trial.user_attrs['config']

best_conf_df = pd.DataFrame(best_configs)
best_conf_df.index.name = 'parameter'

best_conf_df.to_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv', index=True)

# best_conf_df= pd.read_csv(f'assets/results/{data_name},{group},{EXPERIMENT}.csv').set_index('parameter')
# best_conf_df['AutoNHITS']['learning_rate']
# best_conf_df['AutoDeepNPTS']['learning_rate']
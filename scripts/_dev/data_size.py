import os
import warnings

import pandas as pd
from src.loaders import ChronosDataset, LongHorizonDatasetR

warnings.filterwarnings('ignore')

os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

DATASETS = [
    'monash_m1_monthly',
    'monash_m1_quarterly',
    'monash_m3_monthly',
    'monash_m3_quarterly',
    'monash_tourism_monthly',
    'monash_tourism_quarterly',
    'monash_hospital',
    "ECL",
    "Exchange",
    "TrafficL",
    "Weather",
]

LH = ["ECL",
      "Exchange",
      "TrafficL",
      "Weather", ]

d = []
for target in DATASETS:
    if target in LH:
        df, horizon, input_size, freq, seas_len = LongHorizonDatasetR.load_everything(target, resample_to='D')
    else:
        df, horizon, input_size, freq, seas_len = ChronosDataset.load_everything(target)

    d.append(
        {
            'dataset': target,
            'horizon': horizon,
            'input_size': input_size,
            "n_obs": df.shape[0],
            "n_series": len(df['unique_id'].value_counts()),
        }
    )

print(pd.DataFrame(d).to_latex())

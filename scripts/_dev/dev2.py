import pandas as pd
import numpy as np

n, freq, seas_l = 3, 'QE', 4
date_range = pd.date_range(start='2000-01-01', periods=n, freq=freq)
y_values = np.random.randn(n)

df1 = pd.DataFrame({
    'unique_id': ['X'] * n,
    'ds': date_range,
    'y': y_values
})

df2 = pd.DataFrame({
    'unique_id': ['Y'] * n,
    'ds': date_range,
    'y': y_values
})

df3 = pd.DataFrame({
    'unique_id': ['X'] * n,
    'ds': date_range,
    'y': y_values
})

df4 = pd.DataFrame({
    'unique_id': ['X2'] * 2,
    'ds': date_range[:2],
    'y': y_values[:2]
})

df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)
df['fold'] = 0

df_grouped = df.groupby(['fold', 'unique_id']).ngroup()
change_points = df_grouped != df_grouped.shift(1)
groups = change_points.cumsum() - 1

df['unique_id'] = (
    groups.pipe(lambda s: (
            df['unique_id'].astype(str)
            + '_fold' + df['fold'].astype(str)
            + '_x' + s.astype(str)
    ))
)

import numpy as np
import pandas as pd

from utilsforecast.compat import DFType, pl


def _zero_to_nan(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.Series):
        res = series.replace(0, np.nan)
    else:
        res = pl.when(series == 0).then(float("nan")).otherwise(series.abs())
    return res


def mase_scaling_factor(
        seasonality: int,
        train_df: DFType,
        id_col: str = "unique_id",
        target_col: str = "y",
) -> DFType:
    """Mean Absolute Scaled Error (MASE)

    MASE measures the relative prediction
    accuracy of a forecasting method by comparing the mean absolute errors
    of the prediction and the observed value against the mean
    absolute errors of the seasonal naive model.
    The MASE partially composed the Overall Weighted Average (OWA),
    used in the M4 Competition.

    Parameters
    ----------
    seasonality : int
        Main frequency of the time series;
        Hourly 24, Daily 7, Weekly 52, Monthly 12, Quarterly 4, Yearly 1.
    train_df : pandas or polars DataFrame
        Training dataframe with id and actual values. Must be sorted by time.
    id_col : str (default='unique_id')
        Column that identifies each series.
    target_col : str (default='y')
        Column that contains the target.

    Returns
    -------
    pandas or polars Dataframe
        dataframe with one row per id and one column per model.

    References
    ----------
    [1] https://robjhyndman.com/papers/mase.pdf
    """

    lagged = train_df.groupby(id_col, observed=True)[target_col].shift(seasonality)
    scale = train_df[target_col].sub(lagged).abs()
    scale = scale.groupby(train_df[id_col], observed=True).mean()

    scale = _zero_to_nan(scale)

    return scale

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


nf = NeuralForecast3(models=models, freq=freq_str,train_uids=['m1','m2','m3','m4'], test_uids=[])
nf.fit(df=estimation_train)
cv=nf.cross_validation(df=estimation_train,val_size=horizon, test_size=None, n_windows=3)

# Pure inference - no retraining
# For h*3 predictions with rolling origin step_size=1
predictions = nf.predict(df=estimation_test, step_size=1, h=horizon*2)
nf.cross_validation()




##
import pickle
import warnings
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Union

import fsspec
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import utilsforecast.processing as ufp
from coreforecast.grouped_array import GroupedArray
from coreforecast.scalers import (
    LocalBoxCoxScaler,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)
from utilsforecast.compat import DataFrame, DFType, Series, pl_DataFrame, pl_Series
from utilsforecast.validation import validate_freq
from neuralforecast.common.enums import ExplainerEnum
from neuralforecast.losses.pytorch import IQLoss, HuberIQLoss
from neuralforecast.models import (
    GRU,
    KAN,
    LSTM,
    MLP,
    NBEATS,
    NHITS,
    RNN,
    SOFTS,
    TCN,
    TFT,
    Autoformer,
    BiTCN,
    DeepAR,
    DeepNPTS,
    DilatedRNN,
    DLinear,
    FEDformer,
    Informer,
    MLPMultivariate,
    NBEATSx,
    NLinear,
    PatchTST,
    RMoK,
    StemGNN,
    TiDE,
    TimeLLM,
    TimeMixer,
    TimesNet,
    TimeXer,
    TSMixer,
    TSMixerx,
    VanillaTransformer,
    iTransformer,
    xLSTM,
)
from neuralforecast.tsdataset import (
    LocalFilesTimeSeriesDataset,
    TimeSeriesDataset,
    _FilesDataset,
)
from neuralforecast.utils import (
    PredictionIntervals,
    get_prediction_interval_method,
    level_to_quantiles,
    quantiles_to_level,
)

class NeuralForecast3(NeuralForecast):
    def __init__(self, *args, train_uids, test_uids, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_uids = train_uids
        self.test_uids = test_uids

    def _no_refit_cross_validation(
        self,
        df: Optional[DataFrame],
        static_df: Optional[DataFrame],
        n_windows: int,
        step_size: int,
        val_size: Optional[int],
        test_size: int,
        verbose: bool,
        id_col: str,
        time_col: str,
        target_col: str,
        h: int,
        **data_kwargs,
    ) -> DataFrame:
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        train_df = df[df[id_col].isin(self.train_uids)]
        # test_df = df[df[id_col].isin(self.test_uids)]

        # Process and save new dataset (in self)
        if df is not None:
            validate_freq(df[time_col], self.freq)

            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )

            self.train_dataset, self.train_uids2, self.train_last_dates, self.train_ds = self._prepare_fit(
                df=train_df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            print('self.train_dataset')
            print(self.train_uids2)
        else:
            if verbose:
                print("Using stored dataset.")

        if val_size is not None:
            if self.dataset.min_size < (val_size + test_size):
                warnings.warn(
                    "Validation and test sets are larger than the shorter time-series."
                )

        fcsts_df = ufp.cv_times(
            times=self.ds,
            uids=self.uids,
            indptr=self.dataset.indptr,
            h=h,
            test_size=test_size,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
        )
        # the cv_times is sorted by window and then id
        fcsts_df = ufp.sort(fcsts_df, [id_col, "cutoff", time_col])

        fcsts_list: List = []
        for model in self.models:
            if self._add_level and (
                model.loss.outputsize_multiplier > 1
                or isinstance(model.loss, (IQLoss, HuberIQLoss))
            ):
                continue

            model.fit(dataset=self.train_dataset, val_size=val_size, test_size=test_size)
            model_fcsts = model.predict(
                self.dataset, step_size=step_size, h=h, **data_kwargs
            )
            # Append predictions in memory placeholder
            fcsts_list.append(model_fcsts)

        fcsts = np.concatenate(fcsts_list, axis=-1)
        # we may have allocated more space than needed
        # each serie can produce at most (serie.size - 1) // self.h CV windows
        effective_sizes = ufp.counts_by_id(fcsts_df, id_col)["counts"].to_numpy()
        needs_trim = effective_sizes.sum() != fcsts.shape[0]
        if self.scalers_ or needs_trim:
            indptr = np.arange(
                0,
                n_windows * h * (self.dataset.n_groups + 1),
                n_windows * h,
                dtype=np.int32,
            )
            if self.scalers_:
                fcsts = self._scalers_target_inverse_transform(fcsts, indptr)
            if needs_trim:
                # we keep only the effective samples of each serie from the cv results
                trimmed = np.empty_like(
                    fcsts, shape=(effective_sizes.sum(), fcsts.shape[1])
                )
                cv_indptr = np.append(0, effective_sizes).cumsum(dtype=np.int32)
                for i in range(fcsts.shape[1]):
                    ga = GroupedArray(fcsts[:, i], indptr)
                    trimmed[:, i] = ga._tails(cv_indptr)
                fcsts = trimmed

        self._fitted = True

        # Add predictions to forecasts DataFrame
        cols = self._get_model_names(add_level=self._add_level)
        if isinstance(self.uids, pl_Series):
            fcsts = pl_DataFrame(dict(zip(cols, fcsts.T)))
        else:
            fcsts = pd.DataFrame(fcsts, columns=cols)
        fcsts_df = ufp.horizontal_concat([fcsts_df, fcsts])

        # Add original input df's y to forecasts DataFrame
        return ufp.join(
            fcsts_df,
            df[[id_col, time_col, target_col]],
            how="left",
            on=[id_col, time_col],
        )





class NeuralForecast2(NeuralForecast):
    def __init__(self, *args, train_uids=None, test_uids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_uids = train_uids
        self.test_uids = test_uids

    def cross_validation(
        self,
        df=None,
        static_df=None,
        n_windows=1,
        step_size=1,
        val_size=0,
        test_size=None,
        use_init_models=False,
        verbose=False,
        refit=False,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        prediction_intervals=None,
        level=None,
        quantiles=None,
        h=None,
        **data_kwargs
    ):
        df_input = df
        static_df_input = static_df

        # If user provided subset uids, filter accordingly
        if self.train_uids is not None:
            # Filter train dataframe to only train_uids
            if df is not None:
                train_mask = df[id_col].isin(self.train_uids)
                df_train = df[train_mask].copy()
            else:
                df_train = df
            if static_df is not None:
                static_df_train = static_df[static_df[id_col].isin(self.train_uids)].copy()
            else:
                static_df_train = static_df
        else:
            df_train = df
            static_df_train = static_df

        if self.test_uids is not None:
            # Filter test dataframe to only test_uids
            if df is not None:
                test_mask = df[id_col].isin(self.test_uids)
                df_test = df[test_mask].copy()
            else:
                df_test = df
            if static_df is not None:
                static_df_test = static_df[static_df[id_col].isin(self.test_uids)].copy()
            else:
                static_df_test = static_df
        else:
            df_test = df
            static_df_test = static_df

        # Fit on train_uids
        # For refitting, use only train set to fit, predict on test_uids
        # For standard cross validation, train/test sets are managed inside cross_validation, so some care is needed

        # The original cross_validation method fits and predicts on the same set unless you externally partition,
        # so we wrap the fit & predict phases accordingly.

        if self.train_uids is not None or self.test_uids is not None:
            # Fit only on train_uids
            result = super().cross_validation(
                df=df_train,
                static_df=static_df_train,
                n_windows=n_windows,
                step_size=step_size,
                val_size=val_size,
                test_size=test_size,
                use_init_models=use_init_models,
                verbose=verbose,
                refit=refit,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                prediction_intervals=prediction_intervals,
                level=level,
                quantiles=quantiles,
                h=h,
                **data_kwargs,
            )
            # after fitting (inside cross_validation), use predict() on test_uids
            if self.test_uids is not None:
                # Sometimes, test_uids may not be present in the fit call, so the class may need to reload the test_uids
                # Return the cross-val on train, plus the predictions on test set
                predictions_test = self.predict(
                    df=df_test,
                    static_df=static_df_test,
                    h=h,
                    level=level,
                    quantiles=quantiles,
                )
                return result, predictions_test
            else:
                return result
        else:
            # Standard logic if no train_uids/test_uids restriction
            return super().cross_validation(
                df=df_input,
                static_df=static_df_input,
                n_windows=n_windows,
                step_size=step_size,
                val_size=val_size,
                test_size=test_size,
                use_init_models=use_init_models,
                verbose=verbose,
                refit=refit,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                prediction_intervals=prediction_intervals,
                level=level,
                quantiles=quantiles,
                h=h,
                **data_kwargs,
            )
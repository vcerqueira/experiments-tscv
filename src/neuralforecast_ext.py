import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

import utilsforecast.processing as ufp
from coreforecast.grouped_array import GroupedArray
from utilsforecast.compat import DataFrame, pl_DataFrame, pl_Series
from utilsforecast.validation import validate_freq
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import IQLoss, HuberIQLoss


class NeuralForecast2(NeuralForecast):
    # A version of NeuralForecast where models are fit on a subset of df
    # that contains only the unique_ids from train_uids

    def __init__(self, train_uids: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_uids = train_uids

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

        train_df = df[df[id_col].isin(self.train_uids)].copy()

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

            self.train_dataset, *_ = self._prepare_fit(
                df=train_df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )

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

from pprint import pprint
from typing import Optional, List

import pandas as pd
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae
from neuralforecast.auto import (AutoNBEATS,
                                 AutoTiDE,
                                 AutoNLinear,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoPatchTST,
                                 AutoTFT,
                                 AutoDeepNPTS)
from ray.tune.search.variant_generator import generate_variants

from neuralforecast.models import (KAN,
                                   NBEATS,
                                   TiDE,
                                   NLinear,
                                   MLP,
                                   DLinear,
                                   NHITS,
                                   PatchTST,
                                   TFT,
                                   DeepNPTS)


class BaseModelsConfig:
    AUTO_MODEL_CLASSES = {
        'AutoTFT': AutoTFT,
        'AutoNBEATS': AutoNBEATS,
        'AutoTiDE': AutoTiDE,
        'AutoNLinear': AutoNLinear,
        'AutoKAN': AutoKAN,
        'AutoMLP': AutoMLP,
        'AutoDLinear': AutoDLinear,
        'AutoNHITS': AutoNHITS,
        'AutoDeepNPTS': AutoDeepNPTS,
        'AutoPatchTST': AutoPatchTST,
    }

    MODEL_CLASSES = {
        'AutoKAN': KAN,
        'AutoMLP': MLP,
        'AutoDLinear': DLinear,
        'AutoNHITS': NHITS,
        'AutoDeepNPTS': DeepNPTS,
        'AutoNBEATS': NBEATS,
        'AutoTiDE': TiDE,
        'AutoNLinear': NLinear,
        'AutoTFT': TFT,
        'AutoPatchTST': PatchTST,
    }

    NEED_CPU = [
        # 'AutoDeepNPTS',
        # 'AutoTFT',
        # 'AutoPatchTST',
        # 'AutoTiDE',
        # 'AutoNLinear',
        # 'AutoKAN'
    ]

    @classmethod
    def get_pseudo_auto_nf_models(cls,
                                  horizon: int,
                                  input_size: int,
                                  n_samples: int,
                                  engine: str = 'cpu',
                                  limit_epochs: bool = False,
                                  limit_val_batches: Optional[int] = None):
        """

        example:

        BaseModelsConfig.get_pseudo_auto_nf_models(horizon=12,
                                           input_size=12,
                                           n_samples=10,
                                           try_mps=True)


        """

        models = []
        for mod_name, mod in cls.AUTO_MODEL_CLASSES.items():
            mod.default_config['accelerator'] = engine

            if limit_epochs:
                # for basic tests
                mod.default_config['max_steps'] = 2

            if limit_val_batches is not None:
                mod.default_config['limit_val_batches'] = limit_val_batches

            configs = cls.sample_configs(model_name='',
                                         config=mod.default_config,
                                         horizon=horizon,
                                         n_samples=n_samples)

            for i, conf_ in enumerate(configs):
                pprint(conf_)
                conf_['h'] = horizon
                conf_.pop('loss')
                if 'input_size' not in conf_.keys():
                    conf_['input_size'] = input_size

                    if 'input_size_multiplier' in conf_.keys():
                        conf_['input_size'] = 1 * input_size
                        conf_.pop('input_size_multiplier')

                if conf_['scaler_type'] == 'robust':
                    conf_['scaler_type'] = 'standard'

                mod_name_ = cls.MODEL_CLASSES.get(mod_name).__name__

                model_inst = cls.MODEL_CLASSES.get(mod_name)(
                    **conf_,
                    alias=f'{mod_name_}_{i}'
                )

                models.append(model_inst)

        return models

    @classmethod
    def sample_configs(cls, model_name: str, horizon: int, n_samples: int = 10, config: Optional[None] = None):
        # BaseModelsConfig.sample_configs('AutoTFT', horizon=2, n_samples=5)
        if config is None:
            config_pool = cls.AUTO_MODEL_CLASSES.get(model_name).get_default_config(h=horizon, backend='ray')
        else:
            config_pool = config

        samples = []
        # For a random search space, generate_variants yields once per call; use different seeds
        for seed in range(n_samples):
            gen = generate_variants({"config": config_pool}, random_state=seed)
            resolved_vars, spec = next(gen)
            samples.append(spec["config"])

        return samples

    @staticmethod
    def best_validation_variants(cv: pd.DataFrame, model_names: List[str]):
        err = evaluate(df=cv, models=model_names, metrics=[mae])
        mean_err = err.mean(numeric_only=True)[model_names]

        best_variants = (
            mean_err.groupby(mean_err.index.str.split("_").str[0]).idxmin().tolist()
        )

        return best_variants


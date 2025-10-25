from typing import Optional

from ray import tune
from neuralforecast.auto import (AutoGRU,
                                 AutoKAN,
                                 AutoMLP,
                                 AutoLSTM,
                                 AutoDLinear,
                                 AutoNHITS,
                                 AutoPatchTST,
                                 AutoTFT,
                                 AutoDeepNPTS,
                                 AutoDeepAR,
                                 AutoTCN,
                                 AutoDilatedRNN)




class ModelsConfig:

    @classmethod
    def get_auto_nf_models(cls,
                           horizon: int,
                           n_samples: int,
                           limit_epochs: bool = False,
                           limit_val_batches: Optional[int] = None):

        NEED_CPU = ['AutoGRU',
                    'AutoDeepNPTS',
                    'AutoDeepAR',
                    'AutoLSTM',
                    'AutoKAN',
                    'AutoDilatedRNN',
                    'AutoTCN']

        model_cls = {
            # 'AutoKAN': AutoKAN,
            'AutoMLP': AutoMLP,
            # 'AutoDLinear': AutoDLinear,
            # 'AutoNHITS': AutoNHITS,
            # 'AutoDeepNPTS': AutoDeepNPTS,
            # 'AutoTFT': AutoTFT,
            # 'AutoPatchTST': AutoPatchTST,
            # 'AutoGRU': AutoGRU,
            # 'AutoDeepAR': AutoDeepAR,
            # 'AutoLSTM': AutoLSTM,
            # 'AutoDilatedRNN': AutoDilatedRNN,
            # 'AutoTCN': AutoTCN,
        }

        models = []
        for mod_name, mod in model_cls.items():
            if mod_name in NEED_CPU:
                mod.default_config['accelerator'] = 'cpu'
            else:
                mod.default_config['accelerator'] = 'mps'

            if limit_epochs:
                # mod.default_config['max_steps'] = tune.choice([3, 2])
                mod.default_config['max_steps'] = 20

            if limit_val_batches is not None:
                mod.default_config['limit_val_batches'] = limit_val_batches

            model_instance = mod(
                h=horizon,
                num_samples=n_samples,
                alias=mod_name,
            )

            models.append(model_instance)

        return models

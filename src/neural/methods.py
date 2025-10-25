from typing import Optional

from neuralforecast.losses.pytorch import MAE
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

from neuralforecast.models import (GRU,
                                   KAN,
                                   MLP,
                                   LSTM,
                                   DLinear,
                                   NHITS,
                                   PatchTST,
                                   TFT,
                                   DeepNPTS,
                                   DeepAR,
                                   TCN,
                                   DilatedRNN)


class ModelsConfig:
    AUTO_MODEL_CLASSES = {
        'AutoKAN': AutoKAN,
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

    MODEL_CLASSES = {
        'AutoKAN': KAN,
        'AutoMLP': MLP,
        'AutoDLinear': DLinear,
        'AutoNHITS': NHITS,
        'AutoDeepNPTS': DeepNPTS,
        'AutoTFT': TFT,
        'AutoPatchTST': PatchTST,
        'AutoGRU': GRU,
        'AutoDeepAR': DeepAR,
        'AutoLSTM': LSTM,
        'AutoDilatedRNN': DilatedRNN,
        'AutoTCN': TCN,
    }

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

        models = []
        for mod_name, mod in cls.AUTO_MODEL_CLASSES.items():
            if mod_name in NEED_CPU:
                mod.default_config['accelerator'] = 'cpu'
            else:
                mod.default_config['accelerator'] = 'mps'

            if limit_epochs:
                mod.default_config['max_steps'] = 20

            if limit_val_batches is not None:
                mod.default_config['limit_val_batches'] = limit_val_batches

            model_instance = mod(
                h=horizon,
                num_samples=n_samples,
                alias=mod_name,
                valid_loss=MAE(),
                refit_with_val=True,
            )

            models.append(model_instance)

        return models

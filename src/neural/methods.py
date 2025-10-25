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
                           limit_val_batches: bool = False):

        NEED_CPU = ['AutoGRU',
                    'AutoDeepNPTS',
                    'AutoDeepAR',
                    'AutoLSTM',
                    'AutoKAN',
                    'AutoDilatedRNN',
                    'AutoTCN']

        model_cls = {
            'AutoKAN': AutoKAN,
            'AutoMLP': AutoMLP,
            # 'AutoDLinear': AutoDLinear,
            # 'AutoNHITS': AutoNHITS,
            'AutoDeepNPTS': AutoDeepNPTS,
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

            if limit_val_batches:
                # for M4
                mod.default_config['limit_val_batches'] = 50

            model_instance = mod(
                h=horizon,
                num_samples=n_samples,
                alias=mod_name,
            )

            models.append(model_instance)

        return models

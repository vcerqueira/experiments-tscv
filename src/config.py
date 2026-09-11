import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


DRY_RUN = False

N_FOLDS = 5
SEED = 789
HOLDOUT_TR = 0.7
MC_TR = 0.5
MC_TS = 0.2
KFOLD_N_REPEATS = 2
FOLD_BASED_ERROR = False
USE_MPS = _env_bool("USE_MPS", True)
USE_CUDA = _env_bool("USE_CUDA", False)
ENGINE = "mps" if USE_MPS else ("gpu" if USE_CUDA else "cpu")
STEP_SIZE = 1
OUT_SET_MULTIPLIER = 2
if DRY_RUN:
    N_SAMPLES = 2
    LIMIT_EPOCHS = True
else:
    N_SAMPLES = 20
    LIMIT_EPOCHS = False

HOLDOUT_FOR_OUTSET = 0

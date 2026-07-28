import time
import warnings

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from neuralforecast import NeuralForecast
from neuralforecast.common._base_auto import RayOptions
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import AutoMLP

from src.loaders import ChronosDataset
from src.config import ENGINE

warnings.filterwarnings('ignore')

# Configuration
TARGET_DATASET = 'monash_m3_monthly'
N_SAMPLES = 10   # Number of hyperparameter configs to try
N_UIDS = 30      # Number of time series (subsample for speed)
MAX_STEPS = 2000 # Need many steps to get enough validation checkpoints for pruning


def get_config():
    """Hyperparameter search space."""
    return {
        "input_size_multiplier": [1, 2],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),
        "num_layers": tune.choice([1, 2, 3]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "standard"]),
        "max_steps": MAX_STEPS,
        "batch_size": 32,
        "windows_batch_size": 256,
        "accelerator": ENGINE,
        "random_seed": tune.randint(1, 100),
        # Force more frequent validation checks for better pruning
        "val_check_steps": 50,  # Validate every 50 steps (if supported)
    }


def create_model_with_pruning(horizon: int) -> AutoMLP:
    """AutoMLP with aggressive ASHAScheduler pruning."""
    # Note: NeuralForecast reports to Ray Tune only on validation_end (not every step)
    # So "iterations" = validation epochs, not training steps
    # With max_steps=2000, we might get ~20 validation checkpoints
    scheduler = ASHAScheduler(
        max_t=100,           # Max validation checkpoints (not training steps!)
        grace_period=2,      # Start pruning after 2 validation checkpoints
        reduction_factor=3,
        brackets=1,
    )
    print(f"Created ASHAScheduler: {scheduler}")
    
    return AutoMLP(
        h=horizon,
        config=get_config(),
        num_samples=N_SAMPLES,
        alias="MLP_with_pruning",
        valid_loss=MAE(),
        refit_with_val=True,
        backend="ray",
        ray_options=RayOptions(scheduler=scheduler),
        verbose=True,  # Enable verbose to see tuning progress
    )


def create_model_without_pruning(horizon: int) -> AutoMLP:
    """AutoMLP without any scheduler (all trials run to completion)."""
    print("Creating model WITHOUT scheduler (no pruning)")
    
    return AutoMLP(
        h=horizon,
        config=get_config(),
        num_samples=N_SAMPLES,
        alias="MLP_no_pruning",
        valid_loss=MAE(),
        refit_with_val=True,
        backend="ray",
        ray_options=None,  # No scheduler = no pruning
        verbose=True,
    )


def run_experiment(model, df, freq, horizon, label: str) -> dict:
    """Run hyperparameter search and measure time."""
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    
    nf = NeuralForecast(models=[model], freq=freq)
    
    start_time = time.perf_counter()
    nf.fit(df=df, val_size=horizon)
    elapsed = time.perf_counter() - start_time
    
    # Get the fitted model from NeuralForecast
    fitted_model = nf.models[0]
    
    # Debug: show available attributes
    print(f"\nDebug - Model attributes: {[a for a in dir(fitted_model) if not a.startswith('_') and 'result' in a.lower()]}")
    
    # Try different ways to access results
    n_trials = 0
    iterations = []
    
    # Method 1: model.results (Ray Tune ResultGrid)
    if hasattr(fitted_model, 'results') and fitted_model.results is not None:
        try:
            results_list = list(fitted_model.results)
            n_trials = len(results_list)
            print(f"Found {n_trials} results via model.results")
            for res in results_list:
                # Try different ways to get training_iteration
                it = None
                if hasattr(res, 'metrics') and res.metrics:
                    it = res.metrics.get('training_iteration')
                elif hasattr(res, 'last_result') and res.last_result:
                    it = res.last_result.get('training_iteration')
                if it is not None:
                    iterations.append(it)
        except Exception as e:
            print(f"Error accessing results: {e}")
    
    # Method 2: Check for result_grid
    if n_trials == 0 and hasattr(fitted_model, 'result_grid'):
        try:
            n_trials = len(fitted_model.result_grid)
            print(f"Found {n_trials} results via result_grid")
        except:
            pass
    
    # Method 3: Check _result_grid
    if n_trials == 0 and hasattr(fitted_model, '_result_grid'):
        try:
            n_trials = len(fitted_model._result_grid)
            print(f"Found {n_trials} results via _result_grid")
        except:
            pass
    
    print(f"Iterations found: {iterations}")
    
    return {
        'label': label,
        'time_sec': round(elapsed, 1),
        'n_trials': n_trials,
        'iterations': iterations,
        'avg_iterations': round(sum(iterations) / len(iterations), 1) if iterations else None,
        'min_iterations': min(iterations) if iterations else None,
        'max_iterations': max(iterations) if iterations else None,
    }


def print_results(results: list[dict]):
    """Print comparison results."""
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    for r in results:
        print(f"\n{r['label']}:")
        print(f"  Total time: {r['time_sec']} seconds")
        print(f"  Trials: {r['n_trials']}")
        if r['avg_iterations']:
            print(f"  Iterations per trial: min={r['min_iterations']}, max={r['max_iterations']}, avg={r['avg_iterations']}")
    
    if len(results) == 2:
        t_with = results[0]['time_sec']
        t_without = results[1]['time_sec']
        speedup = t_without / t_with if t_with > 0 else 0
        savings = ((t_without - t_with) / t_without) * 100 if t_without > 0 else 0
        
        print("\n" + "-"*60)
        print("SPEEDUP FROM PRUNING:")
        print("-"*60)
        print(f"  Time WITH pruning:    {t_with}s")
        print(f"  Time WITHOUT pruning: {t_without}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time saved: {savings:.1f}%")


# Load data
print("Loading dataset...")
df, horizon, _, freq, _ = ChronosDataset.load_everything(TARGET_DATASET)
df = ChronosDataset.sample_first_uids(df, n_uid=N_UIDS)
print(f"Using {df['unique_id'].nunique()} series, horizon={horizon}")
print(f"Engine: {ENGINE}")
print(f"N_SAMPLES: {N_SAMPLES}")

# Run experiments
results = []

# WITH pruning
model_with = create_model_with_pruning(horizon)
results.append(run_experiment(model_with, df, freq, horizon, "WITH ASHAScheduler (pruning)"))

# WITHOUT pruning
model_without = create_model_without_pruning(horizon)
results.append(run_experiment(model_without, df, freq, horizon, "WITHOUT pruning"))

# Print comparison
print_results(results)

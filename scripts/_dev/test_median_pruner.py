import os
import warnings
from copy import deepcopy

import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from neuralforecast import NeuralForecast
from neuralforecast.common._base_auto import RayOptions
from neuralforecast.losses.pytorch import MAE
from neuralforecast.auto import AutoMLP

from src.loaders import ChronosDataset
from src.config import ENGINE

warnings.filterwarnings('ignore')


os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

TARGET_DATASET = 'monash_m3_monthly'
N_SAMPLES = 50
MAX_STEPS = 200


def create_test_model(horizon: int, with_scheduler: bool = True) -> AutoMLP:
    """Create an AutoMLP model with or without the MedianStoppingRule."""
    
    config = {
        "input_size_multiplier": [1, 2],
        "h": None,
        "hidden_size": tune.choice([64, 128, 256]),  # Varied to create performance differences
        "num_layers": tune.choice([1, 2, 3]),
        "learning_rate": tune.choice([1e-4, 1e-3, 1e-2]),
        "scaler_type": tune.choice([None, "standard"]),
        "max_steps": MAX_STEPS,
        "batch_size": 32,
        "windows_batch_size": 256,
        "accelerator": ENGINE,
        "random_seed": tune.randint(1, 20),
    }
    
    ray_options = None
    if with_scheduler:
        # MedianStoppingRule params:
        # - time_attr: metric used for time (default: "training_iteration")
        # - grace_period: min iterations before stopping allowed
        # - min_samples_required: min trials to compare against
        # Note: metric and mode are set by NeuralForecast internally, don't pass them here
        ray_options = RayOptions(
            scheduler=MedianStoppingRule(
                time_attr="training_iteration",
                grace_period=20,  # Don't stop before 20 iterations
                min_samples_required=3,  # Need at least 3 trials to compare
            )
        )
    
    return AutoMLP(
        h=horizon,
        config=config,
        num_samples=N_SAMPLES,
        alias="TestMLP",
        valid_loss=MAE(),
        refit_with_val=True,
        backend="ray",
        ray_options=ray_options,
    )


def analyze_trial_results(model: AutoMLP) -> pd.DataFrame:
    """Analyze the trial results to check for pruning."""
    
    results = []
    
    # Access the Ray Tune results through the model
    # After fitting, AutoMLP stores results in model.results (list of Result objects)
    if hasattr(model, 'results') and model.results is not None:
        print(f"\nFound {len(model.results)} trial results")
        
        for i, result in enumerate(model.results):
            trial_info = {'trial_idx': i}
            
            # Ray Tune Result object has .metrics dict
            if hasattr(result, 'metrics') and result.metrics:
                trial_info['loss'] = result.metrics.get('loss')
                trial_info['training_iteration'] = result.metrics.get('training_iteration')
            elif hasattr(result, 'last_result') and result.last_result:
                # Alternative access path
                trial_info['loss'] = result.last_result.get('loss')
                trial_info['training_iteration'] = result.last_result.get('training_iteration')
            else:
                # Try direct attribute access
                trial_info['loss'] = getattr(result, 'loss', None)
                trial_info['training_iteration'] = getattr(result, 'training_iteration', None)
            
            # Get config
            if hasattr(result, 'config'):
                trial_info['max_steps'] = result.config.get('max_steps', MAX_STEPS)
                trial_info['hidden_size'] = result.config.get('hidden_size')
                trial_info['learning_rate'] = result.config.get('learning_rate')
            
            # Check if trial was pruned
            if trial_info.get('training_iteration') is not None:
                expected_max = trial_info.get('max_steps', MAX_STEPS)
                trial_info['was_pruned'] = trial_info['training_iteration'] < expected_max
            else:
                trial_info['was_pruned'] = None
            
            results.append(trial_info)
    
    # Also try to access via ResultGrid if available
    if hasattr(model, '_result_grid') and model._result_grid is not None:
        print("\nAccessing results via ResultGrid...")
        for i, result in enumerate(model._result_grid):
            if i >= len(results):
                results.append({'trial_idx': i})
            results[i]['status'] = getattr(result, 'status', 'unknown')
    
    return pd.DataFrame(results)


def run_test(with_scheduler: bool = True):
    """Run the test and report results."""
    
    scheduler_str = "WITH" if with_scheduler else "WITHOUT"
    print(f"\n{'='*60}")
    print(f"RUNNING TEST {scheduler_str} MedianStoppingRule")
    print(f"{'='*60}")
    
    # Load data
    print("\nLoading dataset...")
    df, horizon, _, freq, _ = ChronosDataset.load_everything(TARGET_DATASET)
    
    # Subsample for faster testing
    df = ChronosDataset.sample_first_uids(df, n_uid=30)
    print(f"Using {df['unique_id'].nunique()} series, horizon={horizon}")
    
    # Create and fit model
    print(f"\nCreating AutoMLP with {N_SAMPLES} samples, max_steps={MAX_STEPS}")
    model = create_test_model(horizon, with_scheduler=with_scheduler)
    
    nf = NeuralForecast(models=[model], freq=freq)
    
    print("\nFitting model (this will run hyperparameter search)...")
    nf.fit(df=df, val_size=horizon)
    
    # Debug: inspect model internals
    inspect_model_internals(model)
    
    # Analyze results
    print("\nAnalyzing trial results...")
    results_df = analyze_trial_results(model)
    
    if not results_df.empty:
        print("\n" + "-"*60)
        print("TRIAL RESULTS:")
        print("-"*60)
        print(results_df[['trial_idx', 'loss', 'training_iteration', 'was_pruned']].to_string(index=False))
        
        # Summary statistics
        n_pruned = results_df['was_pruned'].sum() if 'was_pruned' in results_df else 0
        n_completed = results_df['completed_all_steps'].sum() if 'completed_all_steps' in results_df else 0
        
        print("\n" + "-"*60)
        print("SUMMARY:")
        print("-"*60)
        print(f"Total trials: {len(results_df)}")
        print(f"Trials pruned early: {n_pruned}")
        print(f"Trials completed all {MAX_STEPS} steps: {n_completed}")
        
        if results_df['training_iteration'].notna().any():
            avg_iterations = results_df['training_iteration'].mean()
            min_iterations = results_df['training_iteration'].min()
            max_iterations = results_df['training_iteration'].max()
            print(f"Iterations - min: {min_iterations}, max: {max_iterations}, avg: {avg_iterations:.1f}")
        
        if n_pruned > 0:
            print(f"\n✓ MedianStoppingRule IS WORKING - {n_pruned} trials were stopped early!")
        else:
            print(f"\n⚠ No trials were pruned. This could mean:")
            print("  - All trials performed similarly well")
            print("  - grace_period is too high")
            print("  - Not enough samples to trigger pruning")
    else:
        print("\n⚠ Could not access trial results. Checking alternative methods...")
        
        # Try to access via the study object if available
        if hasattr(model, 'study'):
            print(f"Found study with {len(model.study.trials)} trials")
    
    return results_df


def inspect_model_internals(model: AutoMLP):
    """Debug helper to inspect what's available in the model after fitting."""
    print("\n" + "-"*60)
    print("MODEL INTERNALS (debug):")
    print("-"*60)
    
    attrs_to_check = [
        'results', '_result_grid', 'study', 'tuner', 
        'best_config', 'best_result', '_tune_results'
    ]
    
    for attr in attrs_to_check:
        if hasattr(model, attr):
            val = getattr(model, attr)
            val_type = type(val).__name__
            if val is None:
                print(f"  {attr}: None")
            elif hasattr(val, '__len__'):
                print(f"  {attr}: {val_type} with {len(val)} items")
            else:
                print(f"  {attr}: {val_type}")
        else:
            print(f"  {attr}: (not present)")
    
    # If results exist, show structure of first result
    if hasattr(model, 'results') and model.results and len(model.results) > 0:
        first = model.results[0]
        print(f"\n  First result type: {type(first).__name__}")
        print(f"  First result attrs: {[a for a in dir(first) if not a.startswith('_')][:15]}...")
        
        if hasattr(first, 'metrics'):
            print(f"  First result metrics: {first.metrics}")


# Run test
print("="*60)
print("MEDIAN PRUNER VERIFICATION TEST")
print("="*60)
print(f"Engine: {ENGINE}")
print(f"Dataset: {TARGET_DATASET}")
print(f"N_SAMPLES: {N_SAMPLES}")
print(f"MAX_STEPS: {MAX_STEPS}")

results_with = run_test(with_scheduler=True)


import time
import itertools
import warnings
from typing import List, Dict, Any

import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT

from src.loaders import ChronosDataset
from src.config import ENGINE

warnings.filterwarnings('ignore')

BATCH_SIZES = [32, 64, 128]
WINDOWS_BATCH_SIZES = [128, 256, 512, 1024]

FIXED_PARAMS = {
    "hidden_size": 32,
    "n_head": 2,
    "learning_rate": 1e-3,
    "scaler_type": "standard",
    "max_steps": 500,
    "random_seed": 42,
}

TARGET_DATASET = 'monash_m3_monthly'


def run_tft_benchmark(
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        batch_size: int,
        windows_batch_size: int,
        engine: str = ENGINE,
) -> Dict[str, Any]:
    model = TFT(
        h=horizon,
        input_size=int(horizon * 1.5),
        batch_size=batch_size,
        windows_batch_size=windows_batch_size,
        accelerator=engine,
        **FIXED_PARAMS,
    )

    nf = NeuralForecast(models=[model], freq=freq)

    # Time the fit
    start_time = time.perf_counter()
    nf.fit(df=df)
    fit_time = time.perf_counter() - start_time

    # Time the predict
    start_time = time.perf_counter()
    _ = nf.predict()
    predict_time = time.perf_counter() - start_time

    return {
        'batch_size': batch_size,
        'windows_batch_size': windows_batch_size,
        'fit_time_sec': round(fit_time, 2),
        'predict_time_sec': round(predict_time, 2),
        'total_time_sec': round(fit_time + predict_time, 2),
    }


def run_all_benchmarks(
        df: pd.DataFrame,
        horizon: int,
        freq: str,
        batch_sizes: List[int],
        windows_batch_sizes: List[int],
        n_repeats: int = 3,
) -> pd.DataFrame:
    """Run benchmarks for all combinations of batch parameters."""

    results = []
    combinations = list(itertools.product(batch_sizes, windows_batch_sizes))

    print(
        f"Running {len(combinations)} configurations x {n_repeats} repeats = {len(combinations) * n_repeats} total runs")
    print(f"Engine: {ENGINE}")
    print(f"Dataset: {TARGET_DATASET}")
    print(f"Series count: {df['unique_id'].nunique()}")
    print(f"Horizon: {horizon}")
    print("-" * 60)

    for batch_size, windows_batch_size in combinations:
        print(f"\nTesting batch_size={batch_size}, windows_batch_size={windows_batch_size}")

        run_times = []
        for rep in range(n_repeats):
            print(f"  Run {rep + 1}/{n_repeats}...", end=" ", flush=True)
            result = run_tft_benchmark(
                df=df,
                horizon=horizon,
                freq=freq,
                batch_size=batch_size,
                windows_batch_size=windows_batch_size,
            )
            run_times.append(result)
            print(f"fit={result['fit_time_sec']}s, predict={result['predict_time_sec']}s")

        # Aggregate across repeats
        avg_result = {
            'batch_size': batch_size,
            'windows_batch_size': windows_batch_size,
            'fit_time_sec': round(np.mean([r['fit_time_sec'] for r in run_times]), 2),
            'fit_time_std': round(np.std([r['fit_time_sec'] for r in run_times]), 2),
            'predict_time_sec': round(np.mean([r['predict_time_sec'] for r in run_times]), 2),
            'total_time_sec': round(np.mean([r['total_time_sec'] for r in run_times]), 2),
        }
        results.append(avg_result)

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print summary tables and analysis."""

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS (averaged over repeats)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Pivot table: batch_size vs windows_batch_size
    print("\n" + "=" * 60)
    print("FIT TIME (seconds) - batch_size (rows) x windows_batch_size (cols)")
    print("=" * 60)
    pivot_fit = results_df.pivot(
        index='batch_size',
        columns='windows_batch_size',
        values='fit_time_sec'
    )
    print(pivot_fit)

    print("\n" + "=" * 60)
    print("TOTAL TIME (seconds) - batch_size (rows) x windows_batch_size (cols)")
    print("=" * 60)
    pivot_total = results_df.pivot(
        index='batch_size',
        columns='windows_batch_size',
        values='total_time_sec'
    )
    print(pivot_total)

    # Find fastest/slowest
    fastest = results_df.loc[results_df['total_time_sec'].idxmin()]
    slowest = results_df.loc[results_df['total_time_sec'].idxmax()]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Fastest: batch_size={fastest['batch_size']}, "
          f"windows_batch_size={fastest['windows_batch_size']} "
          f"({fastest['total_time_sec']}s)")
    print(f"Slowest: batch_size={slowest['batch_size']}, "
          f"windows_batch_size={slowest['windows_batch_size']} "
          f"({slowest['total_time_sec']}s)")
    print(f"Speedup: {round(slowest['total_time_sec'] / fastest['total_time_sec'], 2)}x")


df, horizon, _, freq, _ = ChronosDataset.load_everything(TARGET_DATASET)

results_df = run_all_benchmarks(
    df=df,
    horizon=horizon,
    freq=freq,
    batch_sizes=BATCH_SIZES,
    windows_batch_sizes=WINDOWS_BATCH_SIZES,
    n_repeats=3,
)

print_summary(results_df)

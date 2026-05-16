"""
shared-utils/stats.py contains common statistical helpers used across experiments.

These are small, focused functions that come up repeatedly in simulation
work: running experiments, summarising results, and comparing distributions.

Usage (from within any experiment folder):
    import sys; sys.path.insert(0, "../shared-utils")
    from stats import bootstrap_ci, compare_distributions, run_trials
"""

from __future__ import annotations

import time
from typing import Callable, Any

import numpy as np
import pandas as pd


def bootstrap_ci(
    data: list[float] | np.ndarray,
    statistic: Callable = np.mean,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for any statistic.

    Parameters
    data        : observed sample
    statistic   : function to apply to each bootstrap sample (default: mean)
    n_bootstrap : number of bootstrap resamples
    confidence  : confidence level (default: 0.95 to 95% CI)

    Returns
    (lower, upper) confidence interval bounds
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(data)
    samples = [statistic(rng.choice(arr, size=len(arr), replace=True))
               for _ in range(n_bootstrap)]
    alpha = (1 - confidence) / 2
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def run_trials(
    trial_fn: Callable[[], Any],
    n: int,
    label: str = "trials",
    verbose: bool = True,
) -> list[Any]:
    """
    Run a function n times and collect results, with optional progress output.

    Parameters
    trial_fn : callable that takes no arguments and returns a result
    n        : number of trials
    label    : printed description for progress output
    verbose  : whether to print start/finish messages

    Returns
    list of n results
    """
    if verbose:
        print(f"Running {n:,} {label}…", end=" ", flush=True)
    t0 = time.perf_counter()
    results = [trial_fn() for _ in range(n)]
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"done in {elapsed:.2f}s")
    return results


def summary_stats(data: list[float] | np.ndarray, label: str = "value") -> pd.Series:
    """
    Compute a standard set of summary statistics.

    Returns a pandas Series with: mean, std, min, p10, p25, p50, p75, p90, max
    """
    arr = np.asarray(data, dtype=float)
    return pd.Series({
        f"{label}_mean": float(np.mean(arr)),
        f"{label}_std":  float(np.std(arr)),
        f"{label}_min":  float(np.min(arr)),
        f"{label}_p10":  float(np.percentile(arr, 10)),
        f"{label}_p25":  float(np.percentile(arr, 25)),
        f"{label}_p50":  float(np.percentile(arr, 50)),
        f"{label}_p75":  float(np.percentile(arr, 75)),
        f"{label}_p90":  float(np.percentile(arr, 90)),
        f"{label}_max":  float(np.max(arr)),
    })


def compare_distributions(
    groups: dict[str, list[float]],
    metric_label: str = "value",
) -> pd.DataFrame:
    """
    Summarise and compare multiple groups of outcomes.

    Parameters
    groups : dict mapping group name to a list of numeric outcomes
    metric_label : name for the metric column

    Returns
    DataFrame with one row per group, columns for mean, std, CI bounds, etc.
    """
    rows = []
    for name, values in groups.items():
        arr = np.asarray(values, dtype=float)
        lo, hi = bootstrap_ci(arr)
        rows.append({
            "group":       name,
            "n":           len(values),
            "mean":        round(float(np.mean(arr)), 3),
            "std":         round(float(np.std(arr)),  3),
            "ci_lower":    round(lo, 3),
            "ci_upper":    round(hi, 3),
            "median":      round(float(np.median(arr)), 3),
            "p10":         round(float(np.percentile(arr, 10)), 3),
            "p90":         round(float(np.percentile(arr, 90)), 3),
        })
    return pd.DataFrame(rows)

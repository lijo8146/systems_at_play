"""
analyzer.py Monte Carlo wildfire suppression strategy comparison.

Runs N simulated fires per strategy across a range of wind/moisture
conditions and compares outcomes: cells burned, value saved, variance.

This is the analytical heart of the wildfire project showing that
the "best" suppression strategy depends on environmental conditions,
and that proactive strategies have higher variance than reactive ones.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from terrain import Wind
from game import STRATEGIES, run_with_suppression


@dataclass
class MonteCarloConfig:
    n_runs           : int   = 200
    seed             : int   = 0
    wind_dir_mean    : float = 180.0    # degrees
    wind_dir_std     : float = 20.0     # variation across runs
    wind_speed_mean  : float = 5.0      # m/s
    wind_speed_std   : float = 2.0
    moisture_mean    : float = 0.3
    moisture_std     : float = 0.08
    budget_per_turn  : int   = 4
    suppression_start: int   = 3
    ignition_row     : int   = 20
    ignition_col     : int   = 10


def run_monte_carlo(config: MonteCarloConfig) -> pd.DataFrame:
    """
    Run Monte Carlo comparison of all three strategies.

    Returns a DataFrame with one row per (strategy, run):
        strategy, run, wind_dir, wind_speed, moisture,
        cells_burned, value_burned, value_saved, timesteps
    """
    rng  = np.random.default_rng(config.seed)
    rows = []

    # Pre-draw environmental conditions (same across strategies)
    wind_dirs   = rng.normal(config.wind_dir_mean, config.wind_dir_std, config.n_runs)
    wind_speeds = np.clip(
        rng.normal(config.wind_speed_mean, config.wind_speed_std, config.n_runs),
        0.5, 15.0,
    )
    terrain_seeds = rng.integers(1000, 9999, config.n_runs)

    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"  Running {strategy_name} ({config.n_runs} runs)...")
        for i in range(config.n_runs):
            wind = Wind(
                direction_deg=float(wind_dirs[i]),
                speed_ms=float(wind_speeds[i]),
            )
            sim_rng = np.random.default_rng(int(terrain_seeds[i]) + 200)
            result  = run_with_suppression(
                terrain_seed     = int(terrain_seeds[i]),
                ignition_pt      = (config.ignition_row, config.ignition_col),
                wind             = wind,
                strategy_fn      = strategy_fn,
                budget_per_turn  = config.budget_per_turn,
                suppression_start= config.suppression_start,
                rng              = sim_rng,
            )
            rows.append({
                "strategy":     strategy_name,
                "run":          i,
                "wind_dir":     float(wind_dirs[i]),
                "wind_speed":   float(wind_speeds[i]),
                "cells_burned": result.cells_burned,
                "value_burned": result.value_burned,
                "value_saved":  result.value_saved,
                "timesteps":    result.timesteps,
            })

    return pd.DataFrame(rows)


def run_baseline(config: MonteCarloConfig) -> pd.DataFrame:
    """Run the same fires with NO suppression for comparison."""
    from simulation import run_simulation
    from terrain import Terrain
    rng  = np.random.default_rng(config.seed)
    rows = []

    wind_dirs   = rng.normal(config.wind_dir_mean, config.wind_dir_std, config.n_runs)
    wind_speeds = np.clip(
        rng.normal(config.wind_speed_mean, config.wind_speed_std, config.n_runs),
        0.5, 15.0,
    )
    terrain_seeds = rng.integers(1000, 9999, config.n_runs)

    print(f"  Running Baseline/No suppression ({config.n_runs} runs)...")
    for i in range(config.n_runs):
        terrain = Terrain(40, 40, seed=int(terrain_seeds[i]),
                          moisture_mean=config.moisture_mean,
                          moisture_std=config.moisture_std)
        wind    = Wind(direction_deg=float(wind_dirs[i]),
                       speed_ms=float(wind_speeds[i]))
        sim_rng = np.random.default_rng(int(terrain_seeds[i]) + 300)
        result  = run_simulation(
            terrain,
            ignition_pt=(config.ignition_row, config.ignition_col),
            wind=wind, rng=sim_rng,
        )
        rows.append({
            "strategy":     "No suppression",
            "run":          i,
            "wind_dir":     float(wind_dirs[i]),
            "wind_speed":   float(wind_speeds[i]),
            "cells_burned": result.cells_burned,
            "value_burned": result.value_burned,
            "value_saved":  result.value_saved,
            "timesteps":    result.timesteps,
        })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per strategy."""
    return (
        df.groupby("strategy")
        .agg(
            mean_burned  = ("cells_burned", "mean"),
            std_burned   = ("cells_burned", "std"),
            p10_burned   = ("cells_burned", lambda x: x.quantile(0.10)),
            p50_burned   = ("cells_burned", lambda x: x.quantile(0.50)),
            p90_burned   = ("cells_burned", lambda x: x.quantile(0.90)),
            mean_saved   = ("value_saved",  "mean"),
            std_saved    = ("value_saved",  "std"),
            cv_burned    = ("cells_burned", lambda x: x.std()/x.mean()),
        )
        .round(2)
        .reset_index()
    )

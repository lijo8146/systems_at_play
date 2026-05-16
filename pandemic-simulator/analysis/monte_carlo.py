"""
monte_carlo.py — Role effectiveness analysis via repeated simulation.

Run many independent simulations under different role configurations
and compare outcomes statistically. This is the analytical core of the
project: instead of a single run, we ask:

    "How much does each player role improve survival odds, cure speed,
     and outbreak containment across N random seeds?"

Usage
─────
    python -m analysis.monte_carlo          # runs default experiment
    python -m analysis.monte_carlo --n 500  # more samples
"""

from __future__ import annotations

import copy
import time
import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from simulation import (
    build_world_network,
    InfectionSimulator,
    InfectionParams,
    Medic, Scientist, QuarantineSpecialist, Epidemiologist, Dispatcher,
)
from simulation.roles import PlayerRole


MAX_STEPS = 80   # cap each run so simulations don't run forever


@dataclass
class RunResult:
    scenario: str
    outcome: str          # "WIN" or "LOSS"
    steps: int
    cure_progress: float
    total_infected: int
    outbreaks: int


def run_once(
    scenario_name: str,
    roles: list[PlayerRole],
    seed: int,
    params: InfectionParams | None = None,
) -> RunResult:
    """Run a single simulation with the given roles and seed."""
    G, cities = build_world_network()
    sim = InfectionSimulator(G, cities, params=params, seed=seed)
    sim.seed_infection("Beijing", count=5)

    for _ in range(MAX_STEPS):
        done = sim.step(roles=roles)
        if done:
            break

    s = sim.summary()
    return RunResult(
        scenario=scenario_name,
        outcome=s["outcome"],
        steps=s["steps"],
        cure_progress=s["cure_progress"],
        total_infected=s["total_ever_infected"],
        outbreaks=s["outbreaks"],
    )


def run_scenario(
    name: str,
    roles: list[PlayerRole],
    n: int,
    params: InfectionParams | None = None,
) -> list[RunResult]:
    """Run *n* independent simulations for one role configuration."""
    return [run_once(name, roles, seed=i, params=params) for i in range(n)]


def results_to_df(results: list[RunResult]) -> pd.DataFrame:
    return pd.DataFrame([vars(r) for r in results])


def analyse(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-scenario statistics."""
    grp = df.groupby("scenario")
    stats = pd.DataFrame({
        "win_rate_%":         grp["outcome"].apply(lambda s: (s == "WIN").mean() * 100).round(1),
        "mean_steps":         grp["steps"].mean().round(1),
        "median_steps":       grp["steps"].median().round(1),
        "mean_infected":      grp["total_infected"].mean().round(0).astype(int),
        "mean_outbreaks":     grp["outbreaks"].mean().round(2),
        "cure_progress_mean": grp["cure_progress"].mean().round(3),
    }).reset_index()
    return stats.sort_values("win_rate_%", ascending=False)


def build_scenarios() -> dict[str, list[PlayerRole]]:
    """
    Define the role combinations to compare.

    Keys become the scenario labels in output tables and charts.
    """
    return {
        "No roles (baseline)":          [],
        "Scientist only":               [Scientist()],
        "Medic only (Beijing)":         [Medic("Beijing")],
        "Epidemiologist only":          [Epidemiologist()],
        "Dispatcher only":              [Dispatcher()],
        "Scientist + Medic":            [Scientist(), Medic("Beijing")],
        "Scientist + Epidemiologist":   [Scientist(), Epidemiologist()],
        "Scientist + Quarantine":       [Scientist(), QuarantineSpecialist()],
        "Full team (all 4 roles)":      [
            Scientist(), Medic("Beijing"), Epidemiologist(), Dispatcher()
        ],
    }


def run_experiment(n: int = 200, verbose: bool = True) -> pd.DataFrame:
    """
    Run all scenarios and return a combined results DataFrame.

    Parameters
    ----------
    n       : simulations per scenario
    verbose : print progress to stdout
    """
    scenarios = build_scenarios()
    all_results: list[RunResult] = []

    t0 = time.perf_counter()
    for label, roles in scenarios.items():
        if verbose:
            print(f"  Running: {label:<40s} ({n:,} sims)…", end=" ", flush=True)
        results = run_scenario(label, roles, n)
        all_results.extend(results)
        if verbose:
            wins = sum(1 for r in results if r.outcome == "WIN")
            print(f"win rate = {wins / n * 100:.1f} %")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\nCompleted {len(all_results):,} simulations in {elapsed:.1f}s")

    return results_to_df(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandemic role-effectiveness Monte Carlo")
    parser.add_argument("--n", type=int, default=200, help="Simulations per scenario")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Pandemic Simulator — Monte Carlo Role Analysis")
    print(f"{'='*60}\n")

    df = run_experiment(n=args.n, verbose=True)
    stats = analyse(df)

    print(f"\n{'─'*60}")
    print("  Results (sorted by win rate)")
    print(f"{'─'*60}")
    print(stats.to_string(index=False))
    print(f"\nRaw results saved to: monte_carlo_results.csv")
    df.to_csv("monte_carlo_results.csv", index=False)

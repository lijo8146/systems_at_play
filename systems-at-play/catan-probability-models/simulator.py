"""
simulator.py Monte Carlo simulation of Catan resource production.

Why simulate when we can calculate analytically?
Expected value per turn is easy to compute from pip values. But players
care about the *distribution* of outcomes over a whole game, not just the
mean. Questions like:

  - What's the probability of getting 0 ore in the first 10 turns?
  - How much does the robber cost you across a 60-turn game?
  - Which settlement location has the most *consistent* yield?

…require simulating many games, not just computing expectations.

This module runs those simulations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from board import Settlement, SETTLEMENTS, STANDARD_HEXES, top_settlements
from dice import PIP_VALUE


@dataclass
class SimulationConfig:
    n_games: int   = 500
    n_turns: int   = 60         
    # (col, row) if robber is placed
    robber_hex: tuple[int, int] | None = None  
    seed: int      = 42


def roll_dice(rng: np.random.Generator) -> int:
    """Roll two six-sided dice, return the sum."""
    return int(rng.integers(1, 7)) + int(rng.integers(1, 7))


def simulate_settlement(
    settlement: Settlement,
    config: SimulationConfig,
) -> pd.DataFrame:
    """
    Simulate resource production for one settlement over many games.

    Returns a DataFrame with one row per (game, turn), columns:
        game, turn, roll, resource_earned, cumulative_resources
    """
    rng = np.random.default_rng(config.seed)
    rows = []

    # Determine which tile numbers are blocked by the robber
    blocked: set[int | None] = set()
    if config.robber_hex:
        from board import hex_at
        blocked_hex = hex_at(*config.robber_hex)
        if blocked_hex:
            blocked.add(blocked_hex.number)

    active_numbers = {n for n in settlement.tile_numbers if n not in blocked}

    for game in range(config.n_games):
        cumulative = 0
        for turn in range(1, config.n_turns + 1):
            roll = roll_dice(rng)
            earned = 1 if roll in active_numbers else 0
            cumulative += earned
            rows.append({
                "game":                 game,
                "turn":                 turn,
                "roll":                 roll,
                "resource_earned":      earned,
                "cumulative_resources": cumulative,
            })

    return pd.DataFrame(rows)


def compare_settlements(
    settlements: list[Settlement],
    config: SimulationConfig,
) -> pd.DataFrame:
    """
    Simulate multiple settlements and return summary statistics.

    Returns a DataFrame with one row per settlement:
        name, mean_total, std_total, p10, p50, p90, p_zero_streak_5
    """
    rows = []
    for s in settlements:
        df = simulate_settlement(s, config)
        totals = df.groupby("game")["resource_earned"].sum()

        # Probability of a "dry spell": 5+ consecutive turns with 0 resources
        def has_dry_spell(game_df: pd.DataFrame, streak: int = 5) -> bool:
            seq = game_df["resource_earned"].tolist()
            count = 0
            for v in seq:
                count = count + 1 if v == 0 else 0
                if count >= streak:
                    return True
            return False

        drought_games = sum(
            1 for g in range(config.n_games)
            if has_dry_spell(df[df["game"] == g])
        )

        rows.append({
            "vertex":          s.name,
            "tiles":           "+".join(str(n) for n in s.tile_numbers),
            "pip_total":       s.pip_total,
            "ev_per_turn":     round(s.ev, 4),
            "mean_total":      round(totals.mean(), 2),
            "std_total":       round(totals.std(), 2),
            "p10":             round(totals.quantile(0.10), 1),
            "p50":             round(totals.quantile(0.50), 1),
            "p90":             round(totals.quantile(0.90), 1),
            "drought_rate_%":  round(drought_games / config.n_games * 100, 1),
            "resources":       ", ".join(s.resources),
        })

    return pd.DataFrame(rows).sort_values("mean_total", ascending=False)


def robber_impact_analysis(
    settlement: Settlement,
    config: SimulationConfig,
) -> dict[str, float]:
    """
    Compare settlement production with and without the robber on each tile.

    Returns a dict mapping tile_number → expected_loss_per_game.
    """
    baseline = simulate_settlement(settlement, config)
    baseline_mean = baseline.groupby("game")["resource_earned"].sum().mean()

    impact = {}
    for hex_obj in settlement.hexes:
        if hex_obj.number is None:
            continue
        robber_config = SimulationConfig(
            n_games=config.n_games,
            n_turns=config.n_turns,
            robber_hex=(hex_obj.col, hex_obj.row),
            seed=config.seed,
        )
        blocked_sim = simulate_settlement(settlement, robber_config)
        blocked_mean = blocked_sim.groupby("game")["resource_earned"].sum().mean()
        impact[hex_obj.number] = round(baseline_mean - blocked_mean, 3)

    return impact


if __name__ == "__main__":
    config = SimulationConfig(n_games=300, n_turns=50)
    top = top_settlements(8)

    print("Comparing top 8 settlement positions (300 simulated games each)…\n")
    df = compare_settlements(top, config)
    print(df[["vertex", "tiles", "pip_total", "mean_total", "std_total",
              "p10", "p90", "drought_rate_%"]].to_string(index=False))

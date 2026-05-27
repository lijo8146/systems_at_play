"""
run_analysis.py to run all Catan probability analyses and save plots.

Usage:
    python run_analysis.py
    python run_analysis.py --n-games 500  # more simulation samples
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dice import print_probability_table
from board import top_settlements, print_settlement_table, SETTLEMENTS
from simulator import SimulationConfig, compare_settlements, robber_impact_analysis
from plots import (
    plot_pip_distribution,
    plot_board_heatmap,
    plot_production_comparison,
    plot_robber_impact,
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run(n_games: int = 300) -> None:
    print("  Catan Probability Analysis")
    
    # Dice distribution
    print("\n[1/4] Dice probability distribution")
    print_probability_table()
    fig = plot_pip_distribution(save_path=f"{OUTPUT_DIR}/pip_distribution.png")
    plt.close(fig)
    print(f"  to Saved {OUTPUT_DIR}/pip_distribution.png")

    # Board heatmap
    print("\n[2/4] Settlement EV heatmap")
    print_settlement_table(n=12)
    fig = plot_board_heatmap(save_path=f"{OUTPUT_DIR}/settlement_ev_heatmap.png")
    plt.close(fig)
    print(f"\n  to Saved {OUTPUT_DIR}/settlement_ev_heatmap.png")

    # Production comparison
    print(f"\n[3/4] Simulating resource production ({n_games} games X top 8 settlements)…")
    config = SimulationConfig(n_games=n_games, n_turns=50)
    top = top_settlements(8)
    stats = compare_settlements(top, config)
    print(stats[["vertex", "tiles", "mean_total", "std_total",
                 "p10", "p90", "drought_rate_%"]].to_string(index=False))
    fig = plot_production_comparison(stats, save_path=f"{OUTPUT_DIR}/resource_production.png")
    plt.close(fig)
    print(f"\n  to Saved {OUTPUT_DIR}/resource_production.png")

    # Robber impact
    print("\n[4/4] Robber impact analysis on best settlement…")
    best = top_settlements(1)[0]
    print(f"  Analysing: {best.name} (tiles {'+'.join(str(n) for n in best.tile_numbers)})")
    impact = robber_impact_analysis(best, config)
    print("  Expected loss if robber blocks each tile for the whole game:")
    for number, loss in sorted(impact.items(), key=lambda x: -x[1]):
        print(f"    Tile {number:>2}: −{loss:.2f} resources")
    fig = plot_robber_impact(impact, best.name,
                             save_path=f"{OUTPUT_DIR}/robber_impact.png")
    plt.close(fig)
    print(f"  to Saved {OUTPUT_DIR}/robber_impact.png")

    print(f"\n All outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=300)
    args = parser.parse_args()
    run(n_games=args.n_games)

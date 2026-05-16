"""
run_simulation.py — Run a single pandemic simulation and generate plots.

Usage
─────
    python run_simulation.py                      # default (no roles)
    python run_simulation.py --roles scientist medic
    python run_simulation.py --roles scientist epidemiologist --seed 99
    python run_simulation.py --monte-carlo --n 300

Available roles: medic, scientist, quarantine_specialist, epidemiologist, dispatcher
Note: 'medic' stations in Beijing (outbreak origin) by default.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt

from simulation import build_world_network, InfectionSimulator, InfectionParams, ALL_ROLES, Medic
from simulation.roles import PlayerRole
from visualization import plot_network_state, plot_spread_timeline, plot_role_comparison
from analysis.monte_carlo import run_experiment, analyse


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_roles(role_names: list[str]) -> list[PlayerRole]:
    roles: list[PlayerRole] = []
    for name in role_names:
        cls = ALL_ROLES.get(name)
        if cls is None:
            print(f"  Warning: unknown role '{name}', skipping.")
            continue
        # Medic needs a stationed city
        if cls is Medic:
            roles.append(cls("Beijing"))
        else:
            roles.append(cls())
    return roles


def run_single(role_names: list[str], seed: int, max_steps: int = 80) -> None:
    G, cities = build_world_network()
    params = InfectionParams()
    sim = InfectionSimulator(G, cities, params=params, seed=seed)
    sim.seed_infection("Beijing", count=5)

    roles = build_roles(role_names)
    role_label = ", ".join(r.name for r in roles) if roles else "No roles (baseline)"
    print(f"\nScenario : {role_label}")
    print(f"Seed     : {seed}")
    print("─" * 50)

    # Snapshot at step 0
    plot_network_state(
        G, cities, step=0, cure_progress=0.0,
        save_path=f"{OUTPUT_DIR}/network_step_000.png",
    )

    for _ in range(max_steps):
        done = sim.step(roles=roles)
        if done:
            break

    s = sim.summary()
    print(f"Outcome  : {s['outcome']}")
    print(f"Steps    : {s['steps']}")
    print(f"Cure     : {s['cure_progress'] * 100:.1f} %")
    print(f"Infected : {s['total_ever_infected']:,}")
    print(f"Outbreaks: {s['outbreaks']}")
    print(f"Hardest hit: {s['worst_city']}")

    # Final network snapshot
    plot_network_state(
        G, cities,
        step=sim.step_count,
        cure_progress=sim.cure_progress,
        save_path=f"{OUTPUT_DIR}/network_final.png",
    )
    print(f"\nSaved → {OUTPUT_DIR}/network_final.png")

    # Timeline
    city_names = list(cities.keys())
    plot_spread_timeline(
        sim.history, city_names,
        save_path=f"{OUTPUT_DIR}/timeline.png",
    )
    print(f"Saved → {OUTPUT_DIR}/timeline.png")
    plt.close("all")


def run_monte_carlo(n: int) -> None:
    print(f"\nRunning Monte Carlo analysis ({n} sims per scenario)…\n")
    df = run_experiment(n=n, verbose=True)
    stats = analyse(df)

    print(f"\n{'─'*60}")
    print("  Results (sorted by win rate)")
    print(f"{'─'*60}")
    print(stats.to_string(index=False))

    df.to_csv(f"{OUTPUT_DIR}/monte_carlo_results.csv", index=False)
    stats.to_csv(f"{OUTPUT_DIR}/monte_carlo_stats.csv", index=False)

    plot_role_comparison(stats, metric="win_rate_%",
                         save_path=f"{OUTPUT_DIR}/role_comparison_winrate.png")
    plot_role_comparison(stats, metric="mean_infected",
                         save_path=f"{OUTPUT_DIR}/role_comparison_infected.png")
    plt.close("all")

    print(f"\nSaved → {OUTPUT_DIR}/role_comparison_winrate.png")
    print(f"Saved → {OUTPUT_DIR}/role_comparison_infected.png")
    print(f"Saved → {OUTPUT_DIR}/monte_carlo_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pandemic spread simulator")
    parser.add_argument(
        "--roles", nargs="*", default=[],
        choices=list(ALL_ROLES.keys()),
        help="Player roles to activate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=80, help="Step cap per run")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run full Monte Carlo role comparison")
    parser.add_argument("--n", type=int, default=200,
                        help="Simulations per scenario (Monte Carlo only)")
    args = parser.parse_args()

    if args.monte_carlo:
        run_monte_carlo(n=args.n)
    else:
        run_single(role_names=args.roles, seed=args.seed, max_steps=args.max_steps)

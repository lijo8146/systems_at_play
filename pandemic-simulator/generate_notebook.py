"""
generate_notebook.py — Builds simulation_notebook.ipynb programmatically.

Run once to regenerate the notebook:
    python generate_notebook.py
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}

cells = []

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)


# ── Title ─────────────────────────────────────────────────────────────────────

cells.append(md("""\
# 🦠 Pandemic Spread Simulator — Analysis Notebook

A stochastic SIR pandemic model on a weighted city network, inspired by the board game *Pandemic*.

**This notebook covers:**
1. Quick-start: running a single simulation
2. Visualising network state and spread timeline
3. Sensitivity analysis — how much does each parameter matter?
4. Monte Carlo role comparison — which player roles win most often?
5. Key modelling insights

---
*Each "step" represents approximately one week of real-world time.*
"""))

# ── 0. Setup ──────────────────────────────────────────────────────────────────

cells.append(md("## 0. Setup"))

cells.append(code("""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))  # repo root

import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

from simulation import (
    build_world_network,
    InfectionSimulator,
    InfectionParams,
    Medic, Scientist, QuarantineSpecialist, Epidemiologist, Dispatcher,
)
from visualization import plot_network_state, plot_spread_timeline, plot_role_comparison
from analysis.monte_carlo import run_experiment, analyse, build_scenarios

%matplotlib inline
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
print("✅ Imports OK")
"""))

# ── 1. Quick-start ────────────────────────────────────────────────────────────

cells.append(md("""\
## 1. Quick-start: run a single simulation

We seed the outbreak in Beijing with 5 initial cases, then step the simulation
until we either cure the disease or lose control of the pandemic.
"""))

cells.append(code("""\
SEED = 42
MAX_STEPS = 80

G, cities = build_world_network()
params = InfectionParams()   # default parameters

# Activate Scientist + Medic stationed in Beijing
roles = [Scientist(), Medic("Beijing")]

sim = InfectionSimulator(G, cities, params=params, seed=SEED)
sim.seed_infection("Beijing", count=5)

for _ in range(MAX_STEPS):
    done = sim.step(roles=roles)
    if done:
        break

summary = sim.summary()
print(f"Outcome       : {summary['outcome']}")
print(f"Steps taken   : {summary['steps']}")
print(f"Cure progress : {summary['cure_progress']*100:.1f} %")
print(f"Total infected: {summary['total_ever_infected']:,}")
print(f"Outbreaks     : {summary['outbreaks']}")
print(f"Hardest hit   : {summary['worst_city']}")
"""))

# ── 2. Network visualisation ──────────────────────────────────────────────────

cells.append(md("""\
## 2. Visualising the city network

The network snapshot colours each city by its current infection rate
(yellow = low, red = high). Node size is proportional to population.
Edge thickness represents travel volume between cities.
"""))

cells.append(code("""\
# Re-run and capture snapshots at specific steps
def run_capturing_steps(roles, seed, steps_to_capture=(0, 10, 20, 30)):
    G, cities = build_world_network()
    sim = InfectionSimulator(G, cities, params=InfectionParams(), seed=seed)
    sim.seed_infection("Beijing", count=5)
    
    snapshots = {}
    if 0 in steps_to_capture:
        snapshots[0] = (copy.deepcopy(cities), sim.cure_progress)
    
    for step in range(1, MAX_STEPS + 1):
        done = sim.step(roles=roles)
        if step in steps_to_capture:
            snapshots[step] = (copy.deepcopy(cities), sim.cure_progress)
        if done:
            break
    
    return snapshots, G

steps_to_capture = (0, 10, 20, 31)
snapshots, G_ref = run_capturing_steps([Scientist(), Medic("Beijing")], seed=SEED,
                                        steps_to_capture=steps_to_capture)

fig, axes = plt.subplots(1, len(snapshots), figsize=(18, 4))
for ax, (step, (cities_snap, cure)) in zip(axes, sorted(snapshots.items())):
    plot_network_state(G_ref, cities_snap, step=step, cure_progress=cure, ax=ax)
plt.suptitle("Network state at selected time steps (Scientist + Medic)", 
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(md("""\
### Baseline (no roles) — the pandemic spirals out of control
"""))

cells.append(code("""\
steps_baseline = (0, 20, 40, 68)
snapshots_base, G_ref = run_capturing_steps([], seed=SEED,
                                             steps_to_capture=steps_baseline)

fig, axes = plt.subplots(1, len(snapshots_base), figsize=(18, 4))
for ax, (step, (cities_snap, cure)) in zip(axes, sorted(snapshots_base.items())):
    plot_network_state(G_ref, cities_snap, step=step, cure_progress=cure, ax=ax)
plt.suptitle("Network state — no roles (LOSS)", 
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
"""))

# ── 3. Spread timeline ────────────────────────────────────────────────────────

cells.append(md("""\
## 3. Spread timeline

Per-city infected curves reveal which cities become outbreak epicentres
and how cure progress races against the spread.
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, (label, roles) in zip(axes, [
    ("Scientist + Medic (WIN)",  [Scientist(), Medic("Beijing")]),
    ("No roles (LOSS)",          []),
]):
    G, cities = build_world_network()
    sim = InfectionSimulator(G, cities, params=InfectionParams(), seed=SEED)
    sim.seed_infection("Beijing", count=5)
    for _ in range(MAX_STEPS):
        if sim.step(roles=roles):
            break

    city_names = list(cities.keys())
    steps = [h["step"] for h in sim.history]
    cmap = plt.cm.tab10

    for i, city in enumerate(city_names):
        vals = [h.get(f"{city}_infected", 0) for h in sim.history]
        ax.plot(steps, vals, label=city, color=cmap(i / len(city_names)), linewidth=1.5)

    ax2 = ax.twinx()
    cure_pct = [h["cure_progress"] * 100 for h in sim.history]
    ax2.plot(steps, cure_pct, color="mediumseagreen", linewidth=2.5,
             linestyle="--", label="Cure %")
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Cure progress (%)", color="mediumseagreen")

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Week (step)")
    ax.set_ylabel("Infected citizens")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

plt.tight_layout()
plt.show()
"""))

# ── 4. Sensitivity analysis ───────────────────────────────────────────────────

cells.append(md("""\
## 4. Sensitivity analysis

Which parameters matter most?

We vary one parameter at a time across a range, running 100 simulations
at each value and recording the win rate and mean outbreak count.
"""))

cells.append(code("""\
def sensitivity_sweep(param_name, values, n_per_value=100, roles=None):
    \"\"\"Vary one InfectionParams field and return a DataFrame of results.\"\"\"
    roles = roles or []
    rows = []
    for val in values:
        kwargs = {param_name: val}
        p = InfectionParams(**kwargs)
        wins, outbreaks_list, infected_list = 0, [], []
        for seed in range(n_per_value):
            G, cities = build_world_network()
            sim = InfectionSimulator(G, cities, params=p, seed=seed)
            sim.seed_infection("Beijing", count=5)
            for _ in range(MAX_STEPS):
                if sim.step(roles=roles):
                    break
            s = sim.summary()
            wins += s["outcome"] == "WIN"
            outbreaks_list.append(s["outbreaks"])
            infected_list.append(s["total_ever_infected"])
        rows.append({
            param_name:       val,
            "win_rate_%":     wins / n_per_value * 100,
            "mean_outbreaks": np.mean(outbreaks_list),
            "mean_infected":  np.mean(infected_list),
        })
    return pd.DataFrame(rows)

print("Running sensitivity sweeps (this takes ~30 seconds)…")

spread_range  = np.arange(0.15, 0.65, 0.05)
travel_range  = np.arange(0.04, 0.36, 0.04)
cure_range    = np.arange(0.006, 0.032, 0.002)

df_spread = sensitivity_sweep("base_spread_rate",    spread_range,  n_per_value=80)
df_travel = sensitivity_sweep("travel_spread_factor", travel_range, n_per_value=80)
df_cure   = sensitivity_sweep("cure_gain_per_step",  cure_range,   n_per_value=80)
print("Done.")
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, df, param, label, color in [
    (axes[0], df_spread, "base_spread_rate",     "Base spread rate (β)",         "#e15759"),
    (axes[1], df_travel, "travel_spread_factor", "Travel spread factor",          "#f28e2b"),
    (axes[2], df_cure,   "cure_gain_per_step",   "Cure gain per step",            "#4e79a7"),
]:
    ax2 = ax.twinx()
    ax.fill_between(df[param], df["win_rate_%"], alpha=0.25, color=color)
    ax.plot(df[param], df["win_rate_%"], color=color, linewidth=2.5, label="Win rate %")
    ax2.plot(df[param], df["mean_outbreaks"], color="grey", linewidth=1.8,
             linestyle="--", label="Mean outbreaks")

    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Win rate (%)", color=color)
    ax.set_ylim(-5, 115)
    ax2.set_ylabel("Mean outbreaks", color="grey")
    ax.set_title(f"Sensitivity: {label}", fontsize=10, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.suptitle("Parameter sensitivity (no roles, 80 runs per point)", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()

print("Key takeaways:")
print(f"  β tipping point  ≈ {df_spread.loc[(df_spread['win_rate_%']-50).abs().idxmin(), 'base_spread_rate']:.2f}")
print(f"  Cure tipping point ≈ {df_cure.loc[(df_cure['win_rate_%']-50).abs().idxmin(), 'cure_gain_per_step']:.3f}")
"""))

# ── 5. Monte Carlo role comparison ────────────────────────────────────────────

cells.append(md("""\
## 5. Monte Carlo role comparison

Run 200 independent simulations for each role configuration.
Random seeds vary across runs to sample the full outcome distribution.
"""))

cells.append(code("""\
print("Running Monte Carlo (200 sims × 9 scenarios)…")
df_mc = run_experiment(n=200, verbose=True)
stats = analyse(df_mc)
stats
"""))

cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_role_comparison(stats, metric="win_rate_%",    ax=axes[0] if False else None)
plot_role_comparison(stats, metric="mean_infected", ax=axes[1] if False else None)

# Replicate inline since plot_role_comparison returns a figure
for ax, metric, title_suffix in [
    (axes[0], "win_rate_%",    "Win rate (%)"),
    (axes[1], "mean_infected", "Mean total citizens infected"),
]:
    df_sorted = stats.sort_values(metric, ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df_sorted)))
    bars = ax.barh(df_sorted["scenario"], df_sorted[metric], color=colors)
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}" if metric == "mean_infected" else f"{val:.1f} %",
                va="center", fontsize=8)
    ax.set_xlabel(title_suffix)
    ax.set_title(f"Role comparison: {title_suffix}", fontweight="bold")
    ax.set_xlim(0, df_sorted[metric].max() * 1.18)

plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# Distribution of outcomes for 3 selected scenarios
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

scenarios_to_plot = ["No roles (baseline)", "Scientist only", "Full team (all 4 roles)"]
for ax, scenario in zip(axes, scenarios_to_plot):
    subset = df_mc[df_mc["scenario"] == scenario]
    wins   = subset[subset["outcome"] == "WIN"]["steps"]
    losses = subset[subset["outcome"] == "LOSS"]["steps"]
    
    ax.hist(wins.values,   bins=15, color="mediumseagreen", alpha=0.7, label=f"WIN  (n={len(wins)})")
    ax.hist(losses.values, bins=15, color="tomato",         alpha=0.7, label=f"LOSS (n={len(losses)})")
    ax.set_xlabel("Steps to terminal state")
    ax.set_ylabel("Count")
    ax.set_title(scenario, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

plt.suptitle("Step-count distribution: WIN vs LOSS by scenario", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
"""))

# ── 6. Key insights ───────────────────────────────────────────────────────────

cells.append(md("""\
## 6. Key modelling insights

### Why cure speed dominates

The Scientist's 2.5× cure multiplier is the single most powerful modifier because
the model is structured as a **race**: cure progress climbs linearly each step
while infection grows (roughly) exponentially in the early phase. Any role that
accelerates cure compresses the race window dramatically.

### Role synergies

The `Scientist + Medic` combo outperforms either alone not because of additive
spread reduction, but because the Medic **buys time** in the epicentre city
(Beijing) during the weeks before the cure reaches 100 %. Less seeding into the
network means fewer outbreak cascades — each of which stochastically delays
global containment.

### The travel-spread tipping point

From the sensitivity sweep, travel spread factor has a **sharp phase transition**:
below ~0.10 the outbreak rarely reaches secondary cities; above ~0.20 it reliably
does. This mirrors real-world findings that international travel bans only matter
above a threshold volume of movement.

### Stochastic variance

Even with a Scientist, there is outcome variance across seeds (visible in the
step-count histograms). The early-game random draw of how many cities get seeded
before containment kicks in explains most of this variance — a reminder that
systems-level interventions reduce *expected* harm but cannot eliminate tail risk.

---
*Next steps: add a latent (Exposed) compartment for SEIR dynamics,
model mutation events mid-simulation, or swap the toy network for 
real IATA flight-volume data.*
"""))

# ── Assemble notebook ─────────────────────────────────────────────────────────

nb.cells = cells

output_path = "/home/claude/pandemic-simulator/simulation_notebook.ipynb"
with open(output_path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {output_path}")

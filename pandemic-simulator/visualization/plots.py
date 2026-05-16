"""
plots.py — Visualisation helpers for the pandemic simulator.

Four main outputs
─────────────────
1. plot_network_state  — snapshot of the city network coloured by infection level
2. plot_spread_timeline — time-series of infected / recovered / cure progress
3. plot_role_comparison — bar chart from Monte Carlo analysis
4. animate_spread       — frame-by-frame GIF of the network evolving (optional)
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
#  Layout — fixed geographic-ish positions for the 10-city network
# ──────────────────────────────────────────────────────────────────────────────

CITY_POSITIONS: dict[str, tuple[float, float]] = {
    "Atlanta":   (-2.0,  0.3),
    "New York":  (-1.2,  0.8),
    "São Paulo": (-1.5, -1.5),
    "London":    ( 0.0,  1.2),
    "Paris":     ( 0.4,  0.9),
    "Lagos":     ( 0.3, -0.8),
    "Cairo":     ( 1.1,  0.4),
    "Beijing":   ( 2.5,  0.9),
    "Tokyo":     ( 3.2,  0.6),
    "Sydney":    ( 3.0, -1.3),
}


def _infection_color(rate: float) -> str:
    """Map 0–1 infection rate to a red-gradient hex colour."""
    cmap = plt.cm.YlOrRd
    return mcolors.to_hex(cmap(min(rate * 2.5, 1.0)))


# ──────────────────────────────────────────────────────────────────────────────
#  1. Network state snapshot
# ──────────────────────────────────────────────────────────────────────────────

def plot_network_state(
    graph: nx.Graph,
    cities: dict,
    step: int,
    cure_progress: float,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Draw the city network coloured by current infection level.

    Node size ∝ city population. Node colour: yellow (low) → red (high).
    Quarantined cities are outlined in blue.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    pos = CITY_POSITIONS

    node_colors  = [_infection_color(cities[n].infection_rate) for n in graph.nodes()]
    node_sizes   = [math.sqrt(cities[n].population) * 0.012 for n in graph.nodes()]
    edge_weights = [graph[u][v]["weight"] for u, v in graph.edges()]

    # Draw edges (thickness ∝ travel weight)
    nx.draw_networkx_edges(
        graph, pos,
        width=[w * 2.5 for w in edge_weights],
        alpha=0.25,
        edge_color="#555555",
        ax=ax,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
    )

    # Quarantine borders
    quarantined = [n for n in graph.nodes() if cities[n].quarantined]
    if quarantined:
        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=quarantined,
            node_color="none",
            edgecolors="royalblue",
            linewidths=3,
            node_size=[s * 1.3 for s in node_sizes],
            ax=ax,
        )

    # Labels with infected counts
    labels = {
        n: f"{n}\n{cities[n].infected:,} inf" for n in graph.nodes()
    }
    nx.draw_networkx_labels(graph, pos, labels, font_size=7, ax=ax)

    # Colour bar legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=mcolors.Normalize(0, 0.4))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Infection rate")

    ax.set_title(
        f"Step {step}  |  Cure progress: {cure_progress * 100:.1f} %",
        fontsize=13, fontweight="bold",
    )
    ax.axis("off")

    if standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  2. Spread timeline
# ──────────────────────────────────────────────────────────────────────────────

def plot_spread_timeline(
    history: list[dict[str, Any]],
    city_names: list[str],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Time-series chart showing per-city infected counts and cure progress.
    """
    steps = [h["step"] for h in history]
    cure  = [h["cure_progress"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Per-city infected curves
    cmap = plt.cm.tab10
    for i, city in enumerate(city_names):
        key = f"{city}_infected"
        vals = [h.get(key, 0) for h in history]
        ax1.plot(steps, vals, label=city, color=cmap(i / len(city_names)), linewidth=1.8)

    ax1.set_ylabel("Infected citizens", fontsize=11)
    ax1.set_title("Infection spread over time", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.7)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Cure progress
    ax2.fill_between(steps, cure, color="mediumseagreen", alpha=0.35)
    ax2.plot(steps, cure, color="seagreen", linewidth=2, label="Cure progress %")
    ax2.axhline(100, color="seagreen", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Cure progress (%)", fontsize=11)
    ax2.set_xlabel("Simulation step (weeks)", fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  3. Role comparison bar chart (Monte Carlo output)
# ──────────────────────────────────────────────────────────────────────────────

def plot_role_comparison(
    stats_df: pd.DataFrame,
    metric: str = "win_rate_%",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing role configurations on a chosen metric.

    Parameters
    ----------
    stats_df : output of analysis.monte_carlo.analyse()
    metric   : column to plot (e.g. 'win_rate_%', 'mean_steps', 'mean_infected')
    """
    df = stats_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df)))
    bars = ax.barh(df["scenario"], df[metric], color=colors, edgecolor="white")

    # Value labels
    for bar, val in zip(bars, df[metric]):
        ax.text(
            bar.get_width() + max(df[metric]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center", fontsize=9,
        )

    label_map = {
        "win_rate_%":    "Win rate (%)",
        "mean_steps":    "Mean steps to terminal state",
        "mean_infected": "Mean total citizens infected",
        "mean_outbreaks":"Mean outbreak events",
    }
    ax.set_xlabel(label_map.get(metric, metric), fontsize=11)
    ax.set_title("Role configuration comparison (Monte Carlo)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(df[metric]) * 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

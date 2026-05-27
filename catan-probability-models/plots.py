"""
plots.py visualisations for Catan probability analysis.

Four main charts
1. pip_distribution      : 2d6 probability curve with pip labels
2. settlement_ev_heatmap : board coloured by settlement EV
3. production_comparison : box plots of simulated resource yields
4. robber_impact         : bar chart of expected loss per blocked tile
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from dice import ROLL_PROBABILITY, PIP_VALUE, RESOURCE_COLORS
from board import STANDARD_HEXES, SETTLEMENTS, Settlement, top_settlements


#  Pip/probability distribution

def plot_pip_distribution(save_path: str | None = None) -> plt.Figure:
    """Bar chart of 2d6 outcomes coloured by pip value."""
    rolls = list(range(2, 13))
    probs = [float(ROLL_PROBABILITY[r]) * 100 for r in rolls]
    pips  = [PIP_VALUE[r] for r in rolls]

    cmap = plt.cm.RdYlGn
    colors = [cmap(p / 6.0) for p in pips]
    colors[5] = "#cc2222"   # 7 = robber — always red

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(rolls, probs, color=colors, edgecolor="white", linewidth=1.5)

    # Pip dots above each bar
    for bar, pip, roll in zip(bars, pips, rolls):
        dots = "●" * pip
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            dots if roll != 7 else "🏴‍☠️",
            ha="center", va="bottom", fontsize=10,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{float(ROLL_PROBABILITY[roll])*100:.1f}%",
            ha="center", va="center", fontsize=8.5, color="white", fontweight="bold",
        )

    ax.set_xticks(rolls)
    ax.set_xlabel("Dice total", fontsize=12)
    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_title("Two-dice probability distribution\n"
                 "Dots represent Catan pip values (pip = ways to roll ÷ 36)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 20)
    ax.axvline(7, color="#cc2222", linestyle="--", alpha=0.4, linewidth=1)
    ax.text(7.15, 18, "Robber", color="#cc2222", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


#  Settlement EV heatmap

def _hex_center(col: int, row: int) -> tuple[float, float]:
    """Convert grid (col, row) to pixel-ish (x, y) for plotting."""
    # Offset rows so the board looks hexagonal
    row_offsets = {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.5, 4: 1.0}
    x = col * 1.1 + row_offsets.get(row, 0)
    y = -row * 0.95
    return x, y


def plot_board_heatmap(save_path: str | None = None) -> plt.Figure:
    """
    Draw the Catan board with hexes coloured by resource type and
    settlement vertices coloured by expected resource value.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Draw hexes as coloured circles
    for h in STANDARD_HEXES:
        x, y = _hex_center(h.col, h.row)
        color = RESOURCE_COLORS.get(h.resource, "#cccccc")
        circle = mpatches.Circle((x, y), 0.48, color=color, alpha=0.75, zorder=1)
        ax.add_patch(circle)
        label = f"{h.label}\n({'·'*h.pip_value})" if h.number else "Desert"
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", zorder=2)

    # Draw settlement vertices coloured by EV
    evs = [s.ev for s in SETTLEMENTS]
    norm = mcolors.Normalize(vmin=min(evs), vmax=max(evs))
    cmap = plt.cm.plasma

    for s in SETTLEMENTS:
        # Average position of adjacent hex centres
        centers = [_hex_center(c, r) for c, r in s.hex_coords]
        sx = np.mean([c[0] for c in centers])
        sy = np.mean([c[1] for c in centers])

        color = cmap(norm(s.ev))
        ax.scatter(sx, sy, c=[color], s=180, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(sx, sy - 0.22, f"{s.ev:.2f}", ha="center", va="top",
                fontsize=6.5, color="white", zorder=4)

    # Colour bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Expected resources/turn", fontsize=10)

    # Legend for resources
    legend_patches = [
        mpatches.Patch(color=c, label=r.capitalize())
        for r, c in RESOURCE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.8)

    ax.set_xlim(-0.8, 5.5)
    ax.set_ylim(-4.4, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Catan board settlement EV heatmap\n"
                 "Hexes: resource type · Dots: settlement expected value (resources/turn)",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


#  Production comparison — box plots

def plot_production_comparison(
    stats_df,          # output of simulator.compare_settlements()
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart comparing mean yield with IQR error bars."""
    df = stats_df.sort_values("mean_total", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(df)))

    y_pos = range(len(df))
    bars = ax.barh(list(y_pos), df["mean_total"], color=colors, alpha=0.85)

    # Error bars from p10 to p90
    ax.errorbar(
        df["mean_total"],
        list(y_pos),
        xerr=[df["mean_total"] - df["p10"], df["p90"] - df["mean_total"]],
        fmt="none", color="white", alpha=0.6, linewidth=1.5, capsize=4,
    )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{row.vertex} ({row.tiles})" for _, row in df.iterrows()],
        fontsize=9,
    )
    ax.set_xlabel("Total resources over 50-turn game (mean ± p10/p90)", fontsize=10)
    ax.set_title("Settlement resource production comparison\n"
                 "Bars: mean Error bars: 10th–90th percentile",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


#  Robber impact

def plot_robber_impact(
    impact: dict[int, float],
    settlement_name: str,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart: expected resource loss per game when robber blocks each tile."""
    numbers = sorted(impact.keys())
    losses  = [impact[n] for n in numbers]
    pips    = [PIP_VALUE[n] for n in numbers]

    cmap = plt.cm.Reds
    colors = [cmap(0.3 + 0.6 * (p / max(pips))) for p in pips]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(n) for n in numbers], losses, color=colors)
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"−{loss:.1f}",
            ha="center", fontsize=9, color="#cc2222", fontweight="bold",
        )

    ax.set_xlabel("Tile number blocked by robber", fontsize=11)
    ax.set_ylabel("Expected resource loss (50-turn game)", fontsize=11)
    ax.set_title(f"Robber impact on settlement {settlement_name}\n"
                 "Expected resources lost if robber stays for the whole game",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

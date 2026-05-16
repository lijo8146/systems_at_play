"""
dice_duel.py non-transitive dice and dominant strategy analysis.

The surprising result
Most people assume dice follow a simple ordering: better dice win more often.
But non-transitive dice break this intuition completely:

    Dice A beats Dice B most of the time.
    Dice B beats Dice C most of the time.
    Dice C beats Dice A most of the time.

This is like rock-paper-scissors: there's no "best" die. The right choice
always depends on what your opponent picks, which is actually a sign of
*good* game balance.

This module:
1. Demonstrates classic non-transitive dice
2. Computes the full win-rate matrix for any set of dice
3. Identifies whether a dominant die exists (bad balance) or not (good balance)
4. Shows how variance interacts with expected value

Run:
    python dice_duel.py
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class Die:
    name: str
    faces: list[int]

    @property
    def mean(self) -> float:
        return sum(self.faces) / len(self.faces)

    @property
    def std(self) -> float:
        return float(np.std(self.faces))

    def __repr__(self) -> str:
        return f"Die({self.name!r}, faces={self.faces}, μ={self.mean:.2f}, σ={self.std:.2f})"


def win_probability(die_a: Die, die_b: Die) -> float:
    """
    Exact probability that die_a beats die_b on a single roll.
    Ties go to neither player.
    """
    wins = sum(
        1
        for a in die_a.faces
        for b in die_b.faces
        if a > b
    )
    total = len(die_a.faces) * len(die_b.faces)
    return wins / total


def build_win_matrix(dice: list[Die]) -> np.ndarray:
    """
    Compute the N×N win-probability matrix.
    Entry [i][j] = P(dice[i] beats dice[j]).
    """
    n = len(dice)
    matrix = np.zeros((n, n))
    for i, j in itertools.product(range(n), range(n)):
        if i != j:
            matrix[i][j] = win_probability(dice[i], dice[j])
    return matrix


def has_dominant_die(matrix: np.ndarray, threshold: float = 0.55) -> int | None:
    """
    Check whether any die wins > threshold of the time against all others.
    Returns the index of the dominant die, or None if balanced.
    """
    n = matrix.shape[0]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        if all(matrix[i][j] > threshold for j in others):
            return i
    return None


def plot_win_matrix(
    dice: list[Die],
    matrix: np.ndarray,
    title: str = "Win probability matrix",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap of win probabilities. Cells > 0.5 = row die wins, < 0.5 = col die wins.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    names = [d.name for d in dice]

    # Custom colormap: red = lose, white = 0.5, green = win
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#d73027", "#ffffbf", "#1a9850"], N=256
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(dice)):
        for j in range(len(dice)):
            val = matrix[i][j]
            text = f"{val:.2f}" if i != j else "—"
            color = "white" if abs(val - 0.5) > 0.2 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(len(dice)))
    ax.set_yticks(range(len(dice)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Opponent die", fontsize=11)
    ax.set_ylabel("Your die", fontsize=11)
    ax.set_title(f"{title}\n(cell = P(row die beats col die))", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Win probability")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def analyse_dice_set(name: str, dice: list[Die]) -> None:
    """Print full analysis for a set of dice."""
    print(f"  {name}")
    for d in dice:
        print(f"  {d.name:>10}: faces={d.faces}  μ={d.mean:.2f}  σ={d.std:.2f}")

    matrix = build_win_matrix(dice)

    print(f"\n  Win-probability matrix (row beats col):")
    header = " " * 14 + "  ".join(f"{d.name:>6}" for d in dice)
    print(header)
    for i, d in enumerate(dice):
        row = "  ".join(
            f"{'—':>6}" if i == j else f"{matrix[i][j]:>6.2f}"
            for j in range(len(dice))
        )
        print(f"  {d.name:>10}:  {row}")

    dominant = has_dominant_die(matrix)
    if dominant is not None:
        print(f"\n    Dominant die found: {dice[dominant].name} : potentially unbalanced!")
    else:
        print(f"\n    No dominant die: these dice are well balanced.")

    return matrix


# Example dice sets

STANDARD = [
    Die("D6-std", [1, 2, 3, 4, 5, 6]),
]

# Efron's non-transitive dice (classic result)
EFRON = [
    Die("Red",    [4, 4, 4, 4, 0, 0]),
    Die("Blue",   [3, 3, 3, 3, 3, 3]),
    Die("Olive",  [6, 6, 2, 2, 2, 2]),
    Die("Purple", [5, 5, 5, 1, 1, 1]),
]

# Unbalanced set (one dominant die)
UNBALANCED = [
    Die("Strong",  [5, 5, 5, 5, 5, 5]),
    Die("Normal",  [1, 2, 3, 4, 5, 6]),
    Die("Weak",    [1, 1, 2, 2, 3, 3]),
]


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs("output", exist_ok=True)

    # Efron non-transitive dice
    m1 = analyse_dice_set("Efron's Non-Transitive Dice", EFRON)
    fig1 = plot_win_matrix(EFRON, m1, "Efron's Non-Transitive Dice",
                           save_path="output/efron_dice.png")
    plt.close(fig1)

    # Unbalanced set
    m2 = analyse_dice_set("Unbalanced Dice Set", UNBALANCED)
    fig2 = plot_win_matrix(UNBALANCED, m2, "Unbalanced Dice Set",
                           save_path="output/unbalanced_dice.png")
    plt.close(fig2)

    print(f"\nSaved to output/efron_dice.png")
    print(f"Saved to output/unbalanced_dice.png")

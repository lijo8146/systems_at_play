"""
dice.py two-dice probability fundamentals for Catan analysis.

The entire strategic framework of Catan rests on the probability
distribution of two six-sided dice. This module makes that
distribution explicit and usable.

"pip value"
Catan's physical tiles use dot patterns (pips) to represent probability
weight. A 6 or 8 tile has 5 pips because there are 5 ways to roll those
numbers. This module formalises that convention.

    pip_value(n) = number of ways to roll n with 2d6  (0 for n=7, desert)
"""

from __future__ import annotations
from fractions import Fraction


# Core lookup tables

# Exact fractional probabilities for every 2d6 outcome
ROLL_PROBABILITY: dict[int, Fraction] = {
    total: Fraction(
        sum(1 for d1 in range(1, 7) for d2 in range(1, 7) if d1 + d2 == total),
        36,
    )
    for total in range(2, 13)
}

# Pip value = numerator of the probability fraction
# (i.e. number of ways to roll that total)
PIP_VALUE: dict[int, int] = {n: int(p * 36) for n, p in ROLL_PROBABILITY.items()}

# Numbers that appear on Catan tiles (7 is the robber, never on a tile)
CATAN_NUMBERS: set[int] = {2, 3, 4, 5, 6, 8, 9, 10, 11, 12}

RESOURCE_COLORS = {
    "ore":    "#8c8c8c",
    "wheat":  "#f5c842",
    "lumber": "#4a7c28",
    "brick":  "#c0522b",
    "wool":   "#a8d878",
    "desert": "#d4c98a",
}


def expected_resources_per_turn(tile_numbers: list[int]) -> float:
    """
    Expected resource cards earned per game turn from a settlement
    adjacent to the given set of tile numbers.

    Each turn one set of dice is rolled. If the roll matches any
    adjacent tile number, the settlement earns one resource card
    (assuming no robber on those tiles).

    Parameters
    ----------
    tile_numbers : list of Catan tile numbers adjacent to a settlement
                   (1–3 values, 7 and None excluded)

    Returns
    float : expected resource cards per turn
    """
    total_ways = sum(PIP_VALUE.get(n, 0) for n in tile_numbers)
    return total_ways / 36.0


def robber_loss_per_turn(tile_number: int, turns_blocked: int = 1) -> float:
    """
    Expected resource loss from the robber sitting on a tile.

    Parameters
    tile_number   : the tile the robber is on
    turns_blocked : how many turns the robber stays there

    Returns
    float : expected resources lost across those turns
    """
    return (PIP_VALUE.get(tile_number, 0) / 36.0) * turns_blocked


def probability_of_at_least_one_resource(tile_numbers: list[int], turns: int) -> float:
    """
    Probability of receiving at least one resource from a settlement
    over a given number of turns.

    Uses the complement rule:
        P(at least one) = 1 - P(none in any turn)^turns
    """
    p_none_per_turn = 1.0 - expected_resources_per_turn(tile_numbers)
    return 1.0 - p_none_per_turn ** turns


def print_probability_table() -> None:
    """Pretty-print the full 2d6 probability table."""
    print(f"{'Roll':>5}  {'Ways':>5}  {'Probability':>12}  {'Pips':>5}  {'%':>7}")
    for n in range(2, 13):
        p = ROLL_PROBABILITY[n]
        pips = PIP_VALUE[n]
        marker = " ← robber" if n == 7 else ""
        print(
            f"{n:>5}  {int(p*36):>5}  {str(p):>12}  "
            f"{'·' * pips:>5}  {float(p)*100:>6.2f}%{marker}"
        )
    print("─" * 45)
    print(f"{'Total':>5}  {'36':>5}  {'36/36':>12}  {'':>5}  {'100.00%':>7}")


if __name__ == "__main__":
    print_probability_table()
    print()
    print("Example: settlement on 6, 8, 9")
    tiles = [6, 8, 9]
    ev = expected_resources_per_turn(tiles)
    print(f"  Expected resources/turn : {ev:.4f}")
    print(f"  P(at least 1 in 5 turns): {probability_of_at_least_one_resource(tiles, 5):.1%}")
    print()
    print("Robber sitting on a 6 for 3 turns:")
    loss = robber_loss_per_turn(6, turns_blocked=3)
    print(f"  Expected loss: {loss:.4f} resources")

"""
board.py standard Catan board model and settlement placement analysis.

Board representation
The 19 Catan hexes are arranged in a 5-row diamond:

    Row 0 (top):     3 hexes
    Row 1:           4 hexes
    Row 2 (middle):  5 hexes
    Row 3:           4 hexes
    Row 4 (bottom):  3 hexes

Each hex is identified by a (col, row) grid position in offset coordinates.

Settlement vertices
A Catan board has 54 settlement vertices. Each vertex is the meeting point
of 2 or 3 hexes. The expected resource value of a settlement at a vertex
equals the sum of pip values of adjacent hexes divided by 36.

This module defines a representative set of ~30 high-interest vertices
covering the full board, sufficient for strategic analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from dice import PIP_VALUE, expected_resources_per_turn, RESOURCE_COLORS


# Hex definition
@dataclass
class Hex:
    """One tile on the Catan board."""
    col: int
    row: int
    resource: str
    number: int | None      

    @property
    def pip_value(self) -> int:
        return PIP_VALUE.get(self.number, 0)

    @property
    def label(self) -> str:
        return str(self.number) if self.number else "D"

    def __repr__(self) -> str:
        return f"Hex({self.resource!r}, #{self.number}, pips={self.pip_value})"


# Standard board layout (Catan base game recommended setup)
# Numbers and resources match the beginner layout from the Catan rulebook.

STANDARD_HEXES: list[Hex] = [
    # Row 0 - 3 hexes
    Hex(0, 0, "ore",    10),
    Hex(1, 0, "wool",    2),
    Hex(2, 0, "lumber",  9),
    # Row 1 - 4 hexes
    Hex(0, 1, "wheat",  12),
    Hex(1, 1, "brick",   6),
    Hex(2, 1, "wool",    4),
    Hex(3, 1, "brick",  10),
    # Row 2 - 5 hexes (middle row)
    Hex(0, 2, "wheat",   9),
    Hex(1, 2, "lumber", 11),
    Hex(2, 2, "desert", None),   # desert (no number)
    Hex(3, 2, "lumber",  3),
    Hex(4, 2, "ore",     8),
    # Row 3 - 4 hexes
    Hex(0, 3, "lumber",  8),
    Hex(1, 3, "ore",     3),
    Hex(2, 3, "wheat",   4),
    Hex(3, 3, "wool",    5),
    # Row 4 - 3 hexes (bottom)
    Hex(0, 4, "brick",   5),
    Hex(1, 4, "wheat",   6),
    Hex(2, 4, "wool",   11),
]


def hex_at(col: int, row: int) -> Hex | None:
    """Look up a hex by grid position."""
    for h in STANDARD_HEXES:
        if h.col == col and h.row == row:
            return h
    return None


# Settlement vertex definition

class Settlement(NamedTuple):
    """
    A named settlement position defined by its adjacent hex positions.

    A real board vertex touches 1–3 hexes. We store each as (col, row)
    and resolve them to actual Hex objects when computing EV.
    """
    name: str
    # list of (col, row) for adjacent hexes
    hex_coords: list[tuple[int, int]]   

    @property
    def hexes(self) -> list[Hex]:
        return [h for c, r in self.hex_coords if (h := hex_at(c, r)) is not None]

    @property
    def tile_numbers(self) -> list[int]:
        return [h.number for h in self.hexes if h.number is not None]

    @property
    def resources(self) -> list[str]:
        return [h.resource for h in self.hexes]

    @property
    def pip_total(self) -> int:
        return sum(PIP_VALUE.get(n, 0) for n in self.tile_numbers)

    @property
    def ev(self) -> float:
        """Expected resources per turn."""
        return expected_resources_per_turn(self.tile_numbers)

    @property
    def resource_diversity(self) -> int:
        """Number of distinct resource types (excluding desert)."""
        return len({r for r in self.resources if r != "desert"})


# Representative settlement vertices covering the full board.
# Named by rough board region for readability.
SETTLEMENTS: list[Settlement] = [
    # Top area
    Settlement("A1", [(0,0),(1,0)]),
    Settlement("A2", [(1,0),(2,0)]),
    Settlement("A3", [(0,0),(1,0),(0,1)]),
    Settlement("A4", [(1,0),(0,1),(1,1)]),
    Settlement("A5", [(1,0),(2,0),(1,1)]),
    Settlement("A6", [(2,0),(1,1),(2,1)]),
    # Upper-middle
    Settlement("B1", [(0,1),(1,1),(0,2)]),
    Settlement("B2", [(1,1),(0,2),(1,2)]),
    Settlement("B3", [(1,1),(2,1),(1,2)]),
    Settlement("B4", [(2,1),(1,2),(2,2)]),
    Settlement("B5", [(2,1),(3,1),(2,2)]),
    Settlement("B6", [(3,1),(2,2),(3,2)]),
    # Middle row (high-value zone)
    Settlement("C1", [(0,2),(1,2),(0,3)]),
    Settlement("C2", [(1,2),(2,2),(1,3)]),
    Settlement("C3", [(1,2),(0,3),(1,3)]),
    Settlement("C4", [(2,2),(3,2),(2,3)]),
    Settlement("C5", [(3,2),(4,2),(3,3)]),
    Settlement("C6", [(3,2),(2,3),(3,3)]),
    # Lower-middle
    Settlement("D1", [(0,3),(1,3),(0,4)]),
    Settlement("D2", [(1,3),(0,4),(1,4)]),
    Settlement("D3", [(1,3),(2,3),(1,4)]),
    Settlement("D4", [(2,3),(1,4),(2,4)]),
    Settlement("D5", [(2,3),(3,3),(2,4)]),
    Settlement("D6", [(3,3),(2,4)]),
    # Bottom
    Settlement("E1", [(0,4),(1,4)]),
    Settlement("E2", [(1,4),(2,4)]),
]


def top_settlements(n: int = 10) -> list[Settlement]:
    """Return the n highest-EV settlements, sorted descending."""
    return sorted(SETTLEMENTS, key=lambda s: s.ev, reverse=True)[:n]


def print_settlement_table(n: int = 15) -> None:
    """Print a ranked table of settlement positions by expected value."""
    ranked = sorted(SETTLEMENTS, key=lambda s: s.ev, reverse=True)
    print(f"\n{'Rank':>4}  {'Vertex':>6}  {'EV/turn':>8}  {'Pips':>5}  {'Tiles':>16}  Resources")
    print("─" * 70)
    for i, s in enumerate(ranked[:n], 1):
        tiles = "+".join(str(n) for n in s.tile_numbers) or "—"
        resources = ", ".join(s.resources)
        print(
            f"{i:>4}  {s.name:>6}  {s.ev:>8.4f}  {s.pip_total:>5}  "
            f"{tiles:>16}  {resources}"
        )

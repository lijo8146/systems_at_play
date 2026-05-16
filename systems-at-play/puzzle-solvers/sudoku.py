"""
sudoku.py is a Sudoku solver with constraint propagation and backtracking.

How it works
1. CONSTRAINT PROPAGATION
   For each empty cell, maintain the set of digits (1–9) that are still
   possible given the current board state (nothing in the same row, column,
   or 3×3 box). Whenever a cell is assigned, update all peers.

2. BACKTRACKING
   When no further progress can be made by propagation alone, pick the
   empty cell with the *fewest remaining possibilities* (minimum remaining
   values heuristic MRV). Try each possibility, recurse, and undo if
   it leads to a dead end.

This combination solves hard puzzles in milliseconds and easy ones instantly.

Board format
Boards are 81-character strings, left to right, top to bottom.
Zeros or dots represent empty cells.

    "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy


# Data types
# cell label to digit (1–9) or None
Board = dict[str, int | None]   

ROWS   = "ABCDEFGHI"
COLS   = "123456789"
DIGITS = set(range(1, 10))

CELLS  = [r + c for r in ROWS for c in COLS]   # "A1" … "I9"

# All 81 peers for each cell: cells in the same row, column, or 3×3 box
def _build_peers() -> dict[str, set[str]]:
    rows  = [[r + c for c in COLS] for r in ROWS]
    cols  = [[r + c for r in ROWS] for c in COLS]
    boxes = [
        [r + c for r in rs for c in cs]
        for rs in ["ABC", "DEF", "GHI"]
        for cs in ["123", "456", "789"]
    ]
    peers: dict[str, set[str]] = {}
    for cell in CELLS:
        peers[cell] = set()
        for unit in rows + cols + boxes:
            if cell in unit:
                peers[cell].update(unit)
        peers[cell].discard(cell)
    return peers

PEERS: dict[str, set[str]] = _build_peers()


# Board parsing

def parse_board(puzzle: str) -> Board:
    """Convert an 81-character string to a Board dict."""
    chars = [c for c in puzzle if c in "0123456789."]
    if len(chars) != 81:
        raise ValueError(f"Expected 81 cells, got {len(chars)}")
    return {
        cell: (int(ch) if ch not in "0." else None)
        for cell, ch in zip(CELLS, chars)
    }


def board_to_string(board: Board) -> str:
    """Render a Board as a formatted 9×9 grid."""
    lines = []
    for i, r in enumerate(ROWS):
        if i in (3, 6):
            lines.append("------+-------+------")
        row_cells = []
        for j, c in enumerate(COLS):
            if j in (3, 6):
                row_cells.append("|")
            val = board[r + c]
            row_cells.append(str(val) if val else ".")
        lines.append(" ".join(row_cells))
    return "\n".join(lines)


# Constraint propagation
# cell to remaining possible digits
Candidates = dict[str, set[int]]   


def initialise_candidates(board: Board) -> Candidates | None:
    """
    Build the initial candidate sets from a parsed board.
    Returns None if the puzzle is immediately contradictory.
    """
    candidates: Candidates = {cell: set(DIGITS) for cell in CELLS}

    for cell, digit in board.items():
        if digit is not None:
            if not assign(candidates, cell, digit):
                return None   # contradiction on given clues

    return candidates


def assign(candidates: Candidates, cell: str, digit: int) -> bool:
    """
    Assign `digit` to `cell` and propagate: remove `digit` from all peers.
    Returns False if this creates a contradiction (any peer has 0 candidates).
    """
    other_values = candidates[cell] - {digit}
    for other in other_values:
        if not eliminate(candidates, cell, other):
            return False
    return True


def eliminate(candidates: Candidates, cell: str, digit: int) -> bool:
    """
    Remove `digit` as a candidate for `cell`.
    Returns False on contradiction.
    """
    # already eliminated, nothing to do
    if digit not in candidates[cell]:
        return True   

    candidates[cell].discard(digit)

    # If a cell has no candidates left, the board is broken
    if len(candidates[cell]) == 0:
        return False

    # If a cell is reduced to one candidate, assign it (naked single)
    if len(candidates[cell]) == 1:
        (sole,) = candidates[cell]
        for peer in PEERS[cell]:
            if not eliminate(candidates, peer, sole):
                return False

    return True


# Backtracking search

def solve(candidates: Candidates) -> Candidates | None:
    """
    Recursive backtracking with MRV (minimum remaining values) heuristic.

    At each step, pick the unsolved cell with the fewest candidates.
    Try each remaining digit; if it leads to a contradiction, undo and try
    the next one.
    """
    # Solved when every cell has exactly one candidate
    if all(len(v) == 1 for v in candidates.values()):
        return candidates

    # MRV: choose the cell with fewest remaining choices (but more than 1)
    cell = min(
        (c for c in CELLS if len(candidates[c]) > 1),
        key=lambda c: len(candidates[c]),
    )

    for digit in candidates[cell]:
        attempt = deepcopy(candidates)
        if assign(attempt, cell, digit):
            result = solve(attempt)
            if result is not None:
                return result

    # all digits failed so backtrack
    return None   


# Public interface

def solve_puzzle(puzzle_str: str) -> tuple[Board | None, float, int]:
    """
    Solve a puzzle from an 81-character string.

    Returns (solved_board, elapsed_seconds, backtrack_count).
    solved_board is None if the puzzle is unsolvable.
    """
    board = parse_board(puzzle_str)
    candidates = initialise_candidates(board)

    t0 = time.perf_counter()
    if candidates is None:
        return None, 0.0, 0

    result = solve(candidates)
    elapsed = time.perf_counter() - t0

    if result is None:
        return None, elapsed, 0

    solved = {cell: next(iter(vals)) for cell, vals in result.items()}
    return solved, elapsed, 0


def print_solution(label: str, puzzle: str) -> None:
    """Solve and pretty-print a puzzle with timing."""
    board = parse_board(puzzle)
    print(f"\n{label}")
    print("─" * 25 + " Input " + "─" * 25)
    print(board_to_string(board))

    solved, elapsed, _ = solve_puzzle(puzzle)
    print("─" * 24 + " Solved " + "─" * 24)
    if solved:
        print(board_to_string(solved))
        print(f"\nSolved in {elapsed*1000:.2f} ms")
    else:
        print("No solution found.")


# Example puzzles

PUZZLES = {
    "Easy": (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    ),
    "Hard (AI Escargot one of the hardest known)": (
        "100007090"
        "030020008"
        "009600500"
        "005300900"
        "010080002"
        "600004000"
        "300000010"
        "040000007"
        "007000300"
    ),
    "Empty (all blanks finds one valid solution)": "0" * 81,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudoku solver")
    parser.add_argument("--puzzle", type=str, default=None,
                        help="81-char puzzle string (0 = empty)")
    args = parser.parse_args()

    if args.puzzle:
        print_solution("Custom puzzle", args.puzzle)
    else:
        for label, puzzle in PUZZLES.items():
            print_solution(label, puzzle)

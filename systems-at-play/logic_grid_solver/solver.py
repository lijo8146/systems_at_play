"""
solver.py logic grid puzzle solver using constraint propagation and backtracking.

A logic grid puzzle assigns N items in each of K categories to N entities
such that all constraints are satisfied.

Representation
candidates[(entity, category)] = set of possible item indices

Algorithm
1. Initialise all cells to full candidate sets
2. Backtrack with MRV (most constrained cell first)
3. At every node, check all clue functions against the current partial
   assignment. If any clue is violated, prune the branch immediately

This is the same propagation and backtracking skeleton as sudoku.py.
The main difference from the original version: clue constraints are checked
at every backtracking node (not just at the fully-solved leaf), which makes
the search fast enough for 5x5 grids.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable


# Data types

@dataclass
class LogicPuzzle:
    n           : int
    categories  : list[str]
    items       : dict[str, list[str]]
    entity_name : str = "entity"
    clues       : list[tuple[str, Callable]] = field(default_factory=list)

    def item_index(self, category: str, item: str) -> int:
        return self.items[category].index(item)

    def cat_index(self, category: str) -> int:
        return self.categories.index(category)


Candidates = dict[tuple[int, int], set[int]]


# Candidate initialisation

def make_candidates(puzzle: LogicPuzzle) -> Candidates:
    return {
        (e, c): set(range(puzzle.n))
        for e in range(puzzle.n)
        for c in range(len(puzzle.categories))
    }

# Constraint propagation

def assign(
    candidates: Candidates,
    entity: int,
    cat: int,
    item: int,
    n: int,
) -> bool:
    """
    Assign item to (entity, cat) and propagate naked singles.
    Returns False on contradiction.
    """
    others = candidates[(entity, cat)] - {item}
    for other in others:
        if not eliminate(candidates, entity, cat, other, n):
            return False
    return True


def eliminate(
    candidates: Candidates,
    entity: int,
    cat: int,
    item: int,
    n: int,
) -> bool:
    """
    Remove item as a candidate for (entity, cat).
    Returns False on contradiction.
    """
    if item not in candidates[(entity, cat)]:
        return True

    candidates[(entity, cat)].discard(item)

    if len(candidates[(entity, cat)]) == 0:
        return False

    # Naked single: propagate to peers in same category
    if len(candidates[(entity, cat)]) == 1:
        (sole,) = candidates[(entity, cat)]
        for other_entity in range(n):
            if other_entity != entity:
                if not eliminate(candidates, other_entity, cat, sole, n):
                    return False

    # Hidden single: if item can only go in one entity for this category
    possible = [e for e in range(n) if item in candidates[(e, cat)]]
    if len(possible) == 0:
        return False
    if len(possible) == 1:
        (sole_entity,) = possible
        if not assign(candidates, sole_entity, cat, item, n):
            return False

    return True


# Clue checking

def check_clues(
    candidates: Candidates,
    puzzle: LogicPuzzle,
) -> bool:
    """
    Check all clue functions against the current (partial) assignment.
    Returns False if any clue is definitively violated.

    Clue functions return True if satisfied OR if not yet determinable
    (some cells still have multiple candidates). They only return False
    when the clue is definitely violated given current assignments.
    """
    partial = {
        k: next(iter(v))
        for k, v in candidates.items()
        if len(v) == 1
    }
    for _, fn in puzzle.clues:
        if not fn(partial, puzzle):
            return False
    return True


# Stub kept for notebook compatibility

def apply_clue_constraints(
    puzzle: LogicPuzzle,
    candidates: Candidates,
) -> tuple[Candidates, int]:
    """
    Lightweight version: just checks clues against current state,
    returns candidates unchanged if consistent.
    (Heavy iterative version was too slow for 5x5 grids with clue checking
    done at every backtracking node in solve() instead.)
    """
    if check_clues(candidates, puzzle):
        return candidates, 0
    return candidates, 0


# Backtracking search

def solve(
    puzzle: LogicPuzzle,
    candidates: Candidates,
    stats: dict | None = None,
) -> Candidates | None:
    """
    Recursive backtracking with MRV heuristic.
    Checks all clues at every node.
    This early pruning makes the search fast enough for 5x5 puzzles.
    """
    if stats is None:
        stats = {"backtracks": 0, "assignments": 0}

    n      = puzzle.n
    n_cats = len(puzzle.categories)

    # Check clues against current partial assignment
    if not check_clues(candidates, puzzle):
        return None

    # Solved when every cell has exactly one candidate
    if all(len(v) == 1 for v in candidates.values()):
        return candidates

    # MRV: cell with fewest candidates (> 1)
    cell = min(
        ((e, c) for e in range(n) for c in range(n_cats)
         if len(candidates[(e, c)]) > 1),
        key=lambda k: len(candidates[k]),
    )
    entity, cat = cell

    for item in sorted(candidates[cell]):
        stats["assignments"] += 1
        attempt = deepcopy(candidates)

        if assign(attempt, entity, cat, item, n):
            result = solve(puzzle, attempt, stats)
            if result is not None:
                return result

        stats["backtracks"] += 1

    return None


# Public interface

def solve_puzzle(
    puzzle: LogicPuzzle,
    verbose: bool = False,
) -> tuple[dict | None, float, dict]:
    """
    Solve a LogicPuzzle. Returns (solution, elapsed_seconds, stats).
    solution maps (entity, cat) to item_index, or None if unsolvable.
    """
    candidates = make_candidates(puzzle)
    stats      = {"backtracks": 0, "assignments": 0}

    t0 = time.perf_counter()
    result = solve(puzzle, candidates, stats)
    elapsed = time.perf_counter() - t0

    if result is None:
        if verbose:
            print("No solution found.")
        return None, elapsed, stats

    solution = {k: next(iter(v)) for k, v in result.items()}

    if verbose:
        solved = sum(1 for v in result.values() if len(v) == 1)
        print(f"Solved {solved}/{puzzle.n * len(puzzle.categories)} cells "
              f"| Backtracks: {stats['backtracks']} "
              f"| Assignments: {stats['assignments']}")

    return solution, elapsed, stats


def format_solution(solution: dict, puzzle: LogicPuzzle) -> str:
    """Render the solution as a formatted table."""
    n      = puzzle.n
    n_cats = len(puzzle.categories)

    col_w    = max(max(len(i) for items in puzzle.items.values()
                       for i in items), 8)
    entity_w = max(len(puzzle.entity_name) + 2, 10)

    header = f"{'':>{entity_w}}  " + "  ".join(
        f"{cat:^{col_w}}" for cat in puzzle.categories
    )
    sep = "─" * len(header)
    lines = [header, sep]

    for e in range(n):
        row_vals = [
            puzzle.items[cat][solution[(e, c)]]
            for c, cat in enumerate(puzzle.categories)
        ]
        label = f"{puzzle.entity_name} {e+1}"
        line  = f"{label:>{entity_w}}  " + "  ".join(
            f"{v:^{col_w}}" for v in row_vals
        )
        lines.append(line)

    return "\n".join(lines)

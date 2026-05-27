"""
game.py wildfire suppression resource allocation game layer.

This module wraps the cellular automaton simulation with a strategic
decision layer: a player allocates suppression resources each turn
to minimize burned area.

Suppression actions
FIREBREAK      : Remove fuel from a cell (fire cannot spread through it).
                 Cost: 2 points. Permanent. Best used proactively.
RETARDANT_DROP : Reduce P(ignition) by 70% in a 3×3 area for 3 timesteps.
                 Cost: 3 points. Limited duration. Best for slowing flanks.
CREW_DEPLOY    : Suppress one BURNING cell (extinguish it immediately).
                 Cost: 1 point. Reactive. Cannot prevent ignition.

Budget
The player receives a fixed budget of suppression points per timestep.
Unspent points do not carry over. This represents the real-world constraint
that suppression resources are limited and must be allocated in real time.

Strategies
REACTIVE       : Spend all points on CREW_DEPLOY at burning cells.
                 Requires no prediction of fire movement.
PROACTIVE      : Spend points on FIREBREAK along projected fire path.
                 Efficient if prediction is correct; wasted if wind shifts.
MIXED          : Firebreaks on primary path and crew on flanks.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from terrain import Terrain, GridCell, FireState, FuelType, FUEL_PROPS, Wind
from simulation import run_simulation, SimResult


# Action types

class ActionType(Enum):
    FIREBREAK       = "firebreak"
    RETARDANT_DROP  = "retardant_drop"
    CREW_DEPLOY     = "crew_deploy"


ACTION_COSTS = {
    ActionType.FIREBREAK:       2,
    ActionType.RETARDANT_DROP:  3,
    ActionType.CREW_DEPLOY:     1,
}

ACTION_DESCRIPTIONS = {
    ActionType.FIREBREAK:      "Remove fuel (permanent barrier, costs 2)",
    ActionType.RETARDANT_DROP: "Reduce spread 70% in 3×3 area for 3 turns (costs 3)",
    ActionType.CREW_DEPLOY:    "Extinguish one burning cell (costs 1)",
}


@dataclass
class SuppressionAction:
    action_type : ActionType
    row         : int
    col         : int


# Expected value of suppression actions

def ev_firebreak(
    terrain    : Terrain,
    row        : int,
    col        : int,
    wind       : Wind,
    turns_remaining: int,
) -> float:
    """
    Expected value of placing a firebreak at (row, col).
    Approximated as: P(fire reaches cell) × cell_value × (cells_protected_downstream).
    """
    cell = terrain.cells[row][col]
    if not cell.is_burnable:
        return 0.0

    # Estimate probability fire reaches this cell
    # (simplified: based on distance from burning cells and wind alignment)
    burning = [
        terrain.cells[r][c]
        for r in range(terrain.rows)
        for c in range(terrain.cols)
        if terrain.cells[r][c].fire_state == FireState.BURNING
    ]
    if not burning:
        return 0.0

    min_dist = min(
        math.sqrt((bc.row - row)**2 + (bc.col - col)**2)
        for bc in burning
    )
    p_reach = max(0.0, 1.0 - min_dist / 15.0)  # decay over ~15 cells

    # Adjust for wind alignment: higher if downwind of burning cells
    wind_boost = np.mean([
        wind.alignment_factor((bc.row, bc.col), (row, col))
        for bc in burning
    ]) if burning else 1.0
    p_reach = min(p_reach * wind_boost, 0.95)

    # Downstream value: cells in wind shadow of this firebreak
    wd_r, wd_c = wind.direction_vec
    downstream_value = 0.0
    for dist in range(1, 8):
        r2 = int(row + wd_r * dist)
        c2 = int(col + wd_c * dist)
        if 0 <= r2 < terrain.rows and 0 <= c2 < terrain.cols:
            downstream_value += terrain.cells[r2][c2].value * (0.8 ** dist)

    own_value = cell.value
    return p_reach * (own_value + downstream_value * 0.5)


def ev_crew_deploy(terrain: Terrain, row: int, col: int) -> float:
    """
    Expected value of extinguishing a burning cell at (row, col).
    Approximated as value of neighbors at risk from this burning cell.
    """
    cell = terrain.cells[row][col]
    if cell.fire_state != FireState.BURNING:
        return 0.0
    neighbor_value = sum(
        nbr.value for nbr in terrain.neighbors(row, col)
        if nbr.is_burnable
    )
    return cell.value * 0.5 + neighbor_value * 0.3


def ev_retardant(terrain: Terrain, row: int, col: int, wind: Wind) -> float:
    """
    Expected value of retardant drop centered at (row, col).
    Sum of EV over 3×3 area.
    """
    total = 0.0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r2, c2 = row + dr, col + dc
            if 0 <= r2 < terrain.rows and 0 <= c2 < terrain.cols:
                total += ev_firebreak(terrain, r2, c2, wind, turns_remaining=5) * 0.4
    return total


# Built-in strategies

StrategyFn = Callable[[Terrain, Wind, int, int], list[SuppressionAction]]


def strategy_reactive(
    terrain: Terrain,
    wind: Wind,
    budget: int,
    timestep: int,
) -> list[SuppressionAction]:
    """
    Reactive strategy: suppress burning cells in order of neighbor risk.
    Spends entire budget on CREW_DEPLOY at the most dangerous burning cells.
    """
    actions = []
    remaining = budget

    burning = sorted(
        [(r, c) for r in range(terrain.rows) for c in range(terrain.cols)
         if terrain.cells[r][c].fire_state == FireState.BURNING],
        key=lambda rc: ev_crew_deploy(terrain, rc[0], rc[1]),
        reverse=True,
    )

    cost = ACTION_COSTS[ActionType.CREW_DEPLOY]
    for r, c in burning:
        if remaining < cost:
            break
        actions.append(SuppressionAction(ActionType.CREW_DEPLOY, r, c))
        remaining -= cost

    return actions


def strategy_proactive(
    terrain: Terrain,
    wind: Wind,
    budget: int,
    timestep: int,
) -> list[SuppressionAction]:
    """
    Proactive strategy: place firebreaks downwind of the fire front.
    Predicts fire path based on current wind direction.
    """
    actions = []
    remaining = budget

    # Find centroid of burning cells
    burning = [
        (r, c) for r in range(terrain.rows) for c in range(terrain.cols)
        if terrain.cells[r][c].fire_state == FireState.BURNING
    ]
    if not burning:
        return actions

    cr = np.mean([r for r, c in burning])
    cc = np.mean([c for r, c in burning])
    wd_r, wd_c = wind.direction_vec

    # Look ahead 5–12 cells downwind for firebreak candidates
    candidates = []
    for dist in range(5, 13):
        r2 = int(cr + wd_r * dist)
        c2 = int(cc + wd_c * dist)
        if 0 <= r2 < terrain.rows and 0 <= c2 < terrain.cols:
            ev = ev_firebreak(terrain, r2, c2, wind, turns_remaining=10)
            candidates.append((ev, r2, c2))

    candidates.sort(reverse=True)
    cost = ACTION_COSTS[ActionType.FIREBREAK]

    for ev, r, c in candidates:
        if remaining < cost:
            break
        if terrain.cells[r][c].is_burnable:
            actions.append(SuppressionAction(ActionType.FIREBREAK, r, c))
            remaining -= cost

    return actions


def strategy_mixed(
    terrain: Terrain,
    wind: Wind,
    budget: int,
    timestep: int,
) -> list[SuppressionAction]:
    """
    Mixed strategy: firebreaks on primary downwind path + crew on flanks.
    Allocates 60% of budget to proactive firebreaks, 40% to reactive crew.
    """
    proactive_budget = int(budget * 0.60)
    reactive_budget  = budget - proactive_budget

    proactive = strategy_proactive(terrain, wind, proactive_budget, timestep)
    reactive  = strategy_reactive(terrain, wind, reactive_budget, timestep)
    return proactive + reactive


STRATEGIES: dict[str, StrategyFn] = {
    "Reactive":   strategy_reactive,
    "Proactive":  strategy_proactive,
    "Mixed":      strategy_mixed,
}


# Apply suppression actions to terrain 

def apply_actions(
    terrain: Terrain,
    actions: list[SuppressionAction],
) -> None:
    """Modify terrain cells according to suppression actions."""
    for action in actions:
        cell = terrain.cells[action.row][action.col]
        if action.action_type == ActionType.FIREBREAK:
            cell.fuel_type  = FuelType.ROAD      # treat as road (no fuel)
            cell.fire_state = FireState.UNBURNED  # ensure it stays unburned
        elif action.action_type == ActionType.RETARDANT_DROP:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r2 = action.row + dr
                    c2 = action.col + dc
                    if 0 <= r2 < terrain.rows and 0 <= c2 < terrain.cols:
                        terrain.cells[r2][c2].retardant = 3
        elif action.action_type == ActionType.CREW_DEPLOY:
            if cell.fire_state == FireState.BURNING:
                cell.fire_state = FireState.SUPPRESSED
                cell.fuel_type  = FuelType.BURNED


# Suppressed simulation run

def run_with_suppression(
    terrain_seed    : int,
    ignition_pt     : tuple[int, int],
    wind            : Wind,
    strategy_fn     : StrategyFn,
    budget_per_turn : int = 4,
    suppression_start: int = 3,
    max_timesteps   : int = 80,
    rng             : np.random.Generator | None = None,
) -> SimResult:
    """
    Run a simulation with active suppression.

    Suppression begins at `suppression_start` timestep, allocating
    `budget_per_turn` points per timestep using `strategy_fn`.
    """
    from terrain import Terrain
    terrain = Terrain(40, 40, seed=terrain_seed,
                      moisture_mean=0.3, moisture_std=0.1)

    if rng is None:
        rng = np.random.default_rng(terrain_seed + 100)

    terrain.ignite(*ignition_pt, timestep=0)
    burn_sequence = [(0, *ignition_pt)]
    perimeter_history = []

    for t in range(1, max_timesteps + 1):
        burning = [
            terrain.cells[r][c]
            for r in range(terrain.rows)
            for c in range(terrain.cols)
            if terrain.cells[r][c].fire_state == FireState.BURNING
        ]

        if not burning:
            break

        perimeter_history.append({(bc.row, bc.col) for bc in burning})

        # Apply suppression before spread
        if t >= suppression_start:
            actions = strategy_fn(terrain, wind, budget_per_turn, t)
            apply_actions(terrain, actions)

        # Spread phase
        new_ignitions = []
        for bc in burning:
            if bc.fire_state != FireState.BURNING:
                continue
            for nbr in terrain.neighbors(bc.row, bc.col):
                if nbr.is_burnable:
                    from simulation import ignition_probability
                    p = ignition_probability(bc, nbr, wind)
                    if rng.random() < p:
                        new_ignitions.append((nbr.row, nbr.col))

        for r, c in new_ignitions:
            if terrain.cells[r][c].is_burnable:
                terrain.cells[r][c].fire_state = FireState.BURNING
                terrain.cells[r][c].burn_time  = t
                burn_sequence.append((t, r, c))

        # Decay phase
        for bc in burning:
            if bc.fire_state == FireState.BURNING and t - bc.burn_time >= 2:
                bc.fire_state = FireState.BURNED
                bc.fuel_type  = FuelType.BURNED

        # Retardant decay
        for row in terrain.cells:
            for cell in row:
                if cell.retardant > 0:
                    cell.retardant -= 1

    cells_burned = sum(
        1 for r in range(terrain.rows) for c in range(terrain.cols)
        if terrain.cells[r][c].fire_state in (FireState.BURNED, FireState.BURNING)
    )
    value_burned = sum(
        FUEL_PROPS[terrain.fuel_map[r][c]].value
        for r in range(terrain.rows) for c in range(terrain.cols)
        if terrain.cells[r][c].fire_state in (FireState.BURNED, FireState.BURNING)
    )
    value_saved = sum(
        terrain.cells[r][c].value
        for r in range(terrain.rows) for c in range(terrain.cols)
        if terrain.cells[r][c].fire_state == FireState.UNBURNED
        and terrain.cells[r][c].is_burnable
    )

    return SimResult(
        terrain=terrain,
        timesteps=t,
        cells_burned=cells_burned,
        value_burned=value_burned,
        value_saved=value_saved,
        burn_sequence=burn_sequence,
        perimeter_history=perimeter_history,
    )

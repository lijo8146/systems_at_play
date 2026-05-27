"""
simulation.py — Cellular automaton wildfire spread engine.

Spread model
------------
Each timestep, every BURNING cell attempts to ignite each burnable neighbor:

    P(ignition) = base_rate[fuel_type]
                × slope_factor(elevation_diff)
                × wind_factor(wind_dir, spread_dir)
                × (1 - moisture)
                × retardant_modifier

This is a simplified version of the physical spread model in FARSITE
(Fire Area Simulator), calibrated to produce realistic spread rates at
moderate wind/moisture conditions.

Key parameters and their physical meaning
------------------------------------------
base_spread_rate : determined by fuel type (grass spreads fastest)
slope_factor     : fire spreads 30-50% faster per 10% slope uphill,
                   slower downhill (simplified version of Rothermel slope factor)
wind_factor      : fire spreads up to 1.8× faster directly downwind,
                   0.3× upwind (dot product alignment with wind vector)
moisture         : at moisture=0 (bone dry) → full spread rate
                   at moisture=1 (saturated) → spread rate → 0

Timestep
--------
Each timestep represents approximately 15 minutes of real fire time
(calibrated so a typical grass fire at moderate wind crosses a 1-hectare
cell in 1–3 timesteps, consistent with NFFL fuel model 1 spread rates).

Simulation terminates when no cells are BURNING (fire extinguished or
all fuel consumed) or max_timesteps is reached.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from terrain import Terrain, GridCell, FireState, FuelType, FUEL_PROPS, Wind


# ── Spread physics ────────────────────────────────────────────────────────────

def slope_factor(from_elev: float, to_elev: float) -> float:
    """
    Rothermel-inspired slope factor.
    Uphill spread: amplified by up to 1.5× at 30° slope.
    Downhill spread: reduced to 0.7× at 30° slope.
    Cell size = 100m; elevation in meters.
    """
    delta_h = to_elev - from_elev
    # tan(slope_angle) approximation over 100m cell
    tan_slope = delta_h / 100.0
    if tan_slope >= 0:
        # Uphill: factor = 1 + 1.67 * phi_s where phi_s = 5.275 * tan^1.3
        phi_s = 5.275 * (abs(tan_slope) ** 1.3)
        return min(1.0 + phi_s * 0.3, 2.5)
    else:
        return max(0.6, 1.0 + tan_slope * 0.5)


def ignition_probability(
    burning_cell: GridCell,
    target_cell:  GridCell,
    wind:         Wind,
    rng:          np.random.Generator | None = None,
) -> float:
    """
    Compute P(ignition) from burning_cell to target_cell in one timestep.
    """
    if not target_cell.is_burnable:
        return 0.0

    base   = FUEL_PROPS[target_cell.fuel_type].base_spread_rate
    slope  = slope_factor(burning_cell.elevation, target_cell.elevation)
    wf     = wind.alignment_factor(
        (burning_cell.row, burning_cell.col),
        (target_cell.row, target_cell.col),
    )
    moist  = 1.0 - target_cell.moisture

    # Retardant effect: reduces effective moisture-adjusted spread
    retardant_mod = 0.3 if target_cell.retardant > 0 else 1.0

    p = base * slope * wf * moist * retardant_mod
    return min(p, 0.95)   # cap at 95% — small random survival chance


# ── Simulation result ─────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """
    Output of a single fire simulation run.

    Attributes
    ----------
    terrain          : final terrain state (fire states updated in place)
    timesteps        : number of timesteps until fire extinguished
    cells_burned     : count of cells that burned
    value_burned     : total weighted value of burned cells
    value_saved      : total weighted value of unburned burnable cells
    burn_sequence    : list of (timestep, row, col) for each ignition event
    perimeter_history: list of cell sets — burning cells at each timestep
    """
    terrain          : Terrain
    timesteps        : int
    cells_burned     : int
    value_burned     : float
    value_saved      : float
    burn_sequence    : list[tuple[int, int, int]]   = field(default_factory=list)
    perimeter_history: list[set[tuple[int,int]]]    = field(default_factory=list)


# ── Main simulation function ───────────────────────────────────────────────────

def run_simulation(
    terrain     : Terrain,
    ignition_pt : tuple[int, int],
    wind        : Wind,
    max_timesteps: int = 80,
    rng         : np.random.Generator | None = None,
) -> SimResult:
    """
    Run the cellular automaton fire spread simulation.

    Parameters
    ----------
    terrain       : Terrain object (modified in place)
    ignition_pt   : (row, col) of initial ignition
    wind          : Wind conditions (fixed for this run)
    max_timesteps : safety cutoff
    rng           : random generator (None → use terrain's rng)

    Returns
    -------
    SimResult with final state and history.
    """
    if rng is None:
        rng = terrain.rng

    burn_sequence = []
    perimeter_history = []

    # Ignite starting cell
    if not terrain.ignite(*ignition_pt, timestep=0):
        # If ignition point is not burnable, find nearest burnable cell
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                r2 = ignition_pt[0] + dr
                c2 = ignition_pt[1] + dc
                if 0 <= r2 < terrain.rows and 0 <= c2 < terrain.cols:
                    if terrain.ignite(r2, c2, timestep=0):
                        break
            else:
                continue
            break

    burn_sequence.append((0, *ignition_pt))

    for t in range(1, max_timesteps + 1):
        # Snapshot of currently burning cells
        burning = [
            terrain.cells[r][c]
            for r in range(terrain.rows)
            for c in range(terrain.cols)
            if terrain.cells[r][c].fire_state == FireState.BURNING
        ]

        if not burning:
            break   # fire extinguished

        perimeter_history.append(
            {(bc.row, bc.col) for bc in burning}
        )

        # Spread phase: attempt ignition of all neighbors
        new_ignitions: list[tuple[int, int]] = []
        for bc in burning:
            for nbr in terrain.neighbors(bc.row, bc.col):
                if nbr.is_burnable:
                    p = ignition_probability(bc, nbr, wind)
                    if rng.random() < p:
                        new_ignitions.append((nbr.row, nbr.col))

        for r, c in new_ignitions:
            if terrain.cells[r][c].is_burnable:
                terrain.cells[r][c].fire_state = FireState.BURNING
                terrain.cells[r][c].burn_time  = t
                burn_sequence.append((t, r, c))

        # Decay phase: BURNING cells that have burned for ≥2 timesteps become BURNED
        for bc in burning:
            if t - bc.burn_time >= 2:
                bc.fire_state = FireState.BURNED
                bc.fuel_type  = FuelType.BURNED

        # Retardant decay
        for r in range(terrain.rows):
            for c in range(terrain.cols):
                cell = terrain.cells[r][c]
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
        terrain           = terrain,
        timesteps         = t,
        cells_burned      = cells_burned,
        value_burned      = value_burned,
        value_saved       = value_saved,
        burn_sequence     = burn_sequence,
        perimeter_history = perimeter_history,
    )


def run_fresh(
    rows: int = 40,
    cols: int = 40,
    seed: int = 42,
    ignition_pt: tuple[int, int] | None = None,
    moisture_mean: float = 0.3,
    moisture_std:  float = 0.1,
    wind_direction: float = 180.0,
    wind_speed: float = 5.0,
    max_timesteps: int = 80,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """
    Convenience wrapper: build terrain, run simulation, return result.
    Ignition defaults to center-left of the grid.
    """
    from terrain import Terrain, Wind
    terrain = Terrain(rows, cols, seed=seed,
                      moisture_mean=moisture_mean,
                      moisture_std=moisture_std)
    if ignition_pt is None:
        ignition_pt = (rows // 2, cols // 4)
    wind = Wind(direction_deg=wind_direction, speed_ms=wind_speed)
    if rng is None:
        rng = np.random.default_rng(seed + 1)
    return run_simulation(terrain, ignition_pt, wind,
                          max_timesteps=max_timesteps, rng=rng)

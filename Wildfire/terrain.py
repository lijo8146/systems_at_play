"""
terrain.py wildfire landscape grid: fuel types, elevation, wind, fire state.

Grid representation
A 2D array of Cell objects on a configurable grid (default 40×40).
Each cell = 1 hectare (100m × 100m).

Fuel types follow a simplified version of the National Fire Danger Rating
System (NFDRS) fuel model classification. Spread rates are calibrated
against published FARSITE benchmarks at moderate wind/moisture conditions.

Elevation is generated via a smoothed random field (summation of sine
waves at multiple frequencies) to produce realistic ridge-and-valley
terrain without requiring real DEM data.

Wind
Wind is represented as a direction (degrees from north, clockwise) and
speed (m/s). It is fixed per simulation run in the base model.
Direction affects spread probability via a dot-product alignment factor:
spread is amplified downwind and suppressed upwind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

import numpy as np


# Enumerations

class FuelType(Enum):
    GRASS   = "grass"     # fastest spread, low intensity
    SHRUB   = "shrub"     # moderate spread, moderate intensity
    FOREST  = "forest"    # slower spread, high intensity, high value
    SLASH   = "slash"     # logging debris  extreme spread rate
    ROCK    = "rock"      # no fuel  natural firebreak
    WATER   = "water"     # no fuel  natural firebreak
    ROAD    = "road"      # no fuel  artificial firebreak
    BURNED  = "burned"    # already consumed no re-ignition


class FireState(Enum):
    UNBURNED    = auto()
    BURNING     = auto()
    BURNED      = auto()
    SUPPRESSED  = auto()   # actively suppressed by crew or retardant


# Fuel properties

@dataclass(frozen=True)
class FuelProperties:
    """Physical fire behavior properties for a fuel type."""
    base_spread_rate : float   # base P(ignition) per timestep, dry conditions
    intensity        : float   # relative fire intensity (0–1); affects suppression cost
    value            : float   # relative value of the cell (scoring weight)
    color            : str     # hex color for visualization


FUEL_PROPS: dict[FuelType, FuelProperties] = {
    FuelType.GRASS:  FuelProperties(0.45, 0.3, 1.0, "#c8b560"),
    FuelType.SHRUB:  FuelProperties(0.35, 0.5, 1.5, "#7a9e4e"),
    FuelType.FOREST: FuelProperties(0.25, 0.8, 3.0, "#2d5a1b"),
    FuelType.SLASH:  FuelProperties(0.55, 0.9, 0.5, "#8b6340"),
    FuelType.ROCK:   FuelProperties(0.00, 0.0, 0.0, "#8c8c8c"),
    FuelType.WATER:  FuelProperties(0.00, 0.0, 0.0, "#2471a3"),
    FuelType.ROAD:   FuelProperties(0.00, 0.0, 0.0, "#555555"),
    FuelType.BURNED: FuelProperties(0.00, 0.0, 0.0, "#2c2c2c"),
}

FIRE_STATE_COLORS = {
    FireState.UNBURNED:   None,          # use fuel color
    FireState.BURNING:    "#e74c3c",
    FireState.BURNED:     "#1a1a1a",
    FireState.SUPPRESSED: "#00bcd4",
}


# Cell

class Cell(NamedTuple):
    row: int
    col: int


@dataclass
class GridCell:
    """State of a single terrain cell."""
    row        : int
    col        : int
    fuel_type  : FuelType
    elevation  : float         # meters
    moisture   : float         # 0.0 = bone dry, 1.0 = fully saturated
    fire_state : FireState = FireState.UNBURNED
    retardant  : int       = 0  # turns of retardant coverage remaining
    burn_time  : int       = -1 # timestep when cell ignited (-1 = never)

    @property
    def is_burnable(self) -> bool:
        return (
            self.fire_state == FireState.UNBURNED
            and FUEL_PROPS[self.fuel_type].base_spread_rate > 0
        )

    @property
    def value(self) -> float:
        return FUEL_PROPS[self.fuel_type].value if self.is_burnable else 0.0

    @property
    def color(self) -> str:
        if self.fire_state != FireState.UNBURNED:
            return FIRE_STATE_COLORS[self.fire_state]
        return FUEL_PROPS[self.fuel_type].color


# Wind

@dataclass
class Wind:
    """
    Wind field (fixed per simulation in base model).

    direction_deg : degrees clockwise from north (0=N, 90=E, 180=S, 270=W)
    speed_ms      : wind speed in m/s (1 m/s ≈ light breeze; 10 m/s = strong wind)
    """
    direction_deg : float = 180.0    # southerly blows north by default
    speed_ms      : float = 5.0

    @property
    def direction_vec(self) -> tuple[float, float]:
        """Unit vector in the direction the wind is blowing (dr, dc)."""
        rad = math.radians(self.direction_deg)
        # north = row decreasing, east = col increasing
        dr = -math.cos(rad)
        dc =  math.sin(rad)
        return (dr, dc)

    def alignment_factor(
        self,
        from_cell: tuple[int, int],
        to_cell:   tuple[int, int],
    ) -> float:
        """
        Dot product alignment between wind direction and spread direction.
        Returns a value in [0, 1+speed_factor]:
          1.0  = spreading crosswind (no wind effect)
          >1.0 = spreading downwind (amplified)
          <1.0 = spreading upwind (suppressed)
        """
        dr = to_cell[0] - from_cell[0]
        dc = to_cell[1] - from_cell[1]
        mag = math.sqrt(dr*dr + dc*dc)
        if mag == 0:
            return 1.0
        dr_n, dc_n = dr/mag, dc/mag
        wd_r, wd_c = self.direction_vec
        dot = dr_n*wd_r + dc_n*wd_c   # -1 to +1
        # Wind speed factor: 10 m/s to up to 1.5× amplification downwind
        speed_factor = min(self.speed_ms / 10.0, 0.8)
        return 1.0 + speed_factor * dot


# Terrain grid

class Terrain:
    """
    A 2D grid of GridCell objects representing the wildfire landscape.

    Parameters
    rows, cols   : grid dimensions (default 40×40)
    seed         : random seed for reproducible terrain generation
    moisture_mean: mean moisture level (0.0–1.0)
    moisture_std : std dev of per-cell moisture variation
    """

    def __init__(
        self,
        rows: int = 40,
        cols: int = 40,
        seed: int = 42,
        moisture_mean: float = 0.3,
        moisture_std:  float = 0.1,
    ):
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.rng  = np.random.default_rng(seed)

        self.elevation = self._generate_elevation()
        self.moisture  = self._generate_moisture(moisture_mean, moisture_std)
        self.fuel_map  = self._generate_fuel_map()
        self.cells     = self._build_cells()

    # Private generation methods

    def _generate_elevation(self) -> np.ndarray:
        """
        Smooth random elevation field via summed sine waves at multiple
        frequencies. Produces realistic ridge-and-valley terrain.
        Range approximately 0–500m.
        """
        elev = np.zeros((self.rows, self.cols))
        freqs_amplitudes = [
            (1, 200), (2, 100), (4, 50), (8, 25), (16, 12),
        ]
        for freq, amp in freqs_amplitudes:
            phase_r = self.rng.uniform(0, 2*math.pi)
            phase_c = self.rng.uniform(0, 2*math.pi)
            for r in range(self.rows):
                for c in range(self.cols):
                    elev[r, c] += amp * (
                        math.sin(freq * r / self.rows * 2*math.pi + phase_r)
                        + math.sin(freq * c / self.cols * 2*math.pi + phase_c)
                    )
        # Normalize to 0–500m
        elev -= elev.min()
        elev  = elev / elev.max() * 500
        return elev

    def _generate_moisture(
        self, mean: float, std: float
    ) -> np.ndarray:
        """Per-cell moisture correlated with elevation (valleys wetter)."""
        base = self.rng.normal(mean, std, (self.rows, self.cols))
        # Low elevation → slightly wetter
        elev_norm = (self.elevation - self.elevation.min()) / (
            (self.elevation.max() - self.elevation.min()) + 1e-9
        )
        base += 0.1 * (1 - elev_norm)
        return np.clip(base, 0.0, 1.0)

    def _generate_fuel_map(self) -> np.ndarray:
        """
        Fuel type array based on elevation bands and random noise.

        Low elevation  : grass / shrub (open areas)
        Mid elevation  : shrub / forest mix
        High elevation : forest / rock
        Water bodies   : randomly placed valleys
        Roads          : two diagonal lines crossing the grid
        """
        fuel = np.full((self.rows, self.cols), FuelType.GRASS)
        elev_norm = (self.elevation - self.elevation.min()) / (
            (self.elevation.max() - self.elevation.min()) + 1e-9
        )

        for r in range(self.rows):
            for c in range(self.cols):
                e = elev_norm[r, c]
                roll = self.rng.random()
                if e > 0.75:
                    fuel[r, c] = FuelType.FOREST if roll > 0.2 else FuelType.ROCK
                elif e > 0.45:
                    fuel[r, c] = FuelType.FOREST if roll > 0.5 else FuelType.SHRUB
                elif e > 0.20:
                    fuel[r, c] = FuelType.SHRUB if roll > 0.4 else FuelType.GRASS
                else:
                    fuel[r, c] = FuelType.GRASS

        # Add water body in lowest valley
        low_r, low_c = np.unravel_index(
            self.elevation.argmin(), self.elevation.shape
        )
        for r in range(max(0, low_r-2), min(self.rows, low_r+3)):
            for c in range(max(0, low_c-2), min(self.cols, low_c+3)):
                if self.rng.random() > 0.3:
                    fuel[r, c] = FuelType.WATER

        # Roads as diagonal strips
        for r in range(self.rows):
            c1 = int(r * self.cols / self.rows)
            c2 = self.cols - 1 - c1
            for dc in [-1, 0, 1]:
                if 0 <= c1+dc < self.cols:
                    if self.rng.random() > 0.15:
                        fuel[r, c1+dc] = FuelType.ROAD
                if 0 <= c2+dc < self.cols and c2 != c1:
                    if self.rng.random() > 0.15:
                        fuel[r, c2+dc] = FuelType.ROAD

        # Scatter some slash (logging debris)
        for _ in range(self.rows * self.cols // 40):
            r = self.rng.integers(0, self.rows)
            c = self.rng.integers(0, self.cols)
            if fuel[r, c] == FuelType.FOREST:
                fuel[r, c] = FuelType.SLASH

        return fuel

    def _build_cells(self) -> list[list[GridCell]]:
        """Instantiate GridCell objects from generated arrays."""
        return [
            [
                GridCell(
                    row       = r,
                    col       = c,
                    fuel_type = self.fuel_map[r, c],
                    elevation = float(self.elevation[r, c]),
                    moisture  = float(self.moisture[r, c]),
                )
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

    # Public interface

    def cell(self, row: int, col: int) -> GridCell:
        return self.cells[row][col]

    def neighbors(self, row: int, col: int) -> list[GridCell]:
        """8-connected neighbors (including diagonals)."""
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r2, c2 = row + dr, col + dc
                if 0 <= r2 < self.rows and 0 <= c2 < self.cols:
                    nbrs.append(self.cells[r2][c2])
        return nbrs

    def ignite(self, row: int, col: int, timestep: int = 0) -> bool:
        """Set a cell to BURNING. Returns False if cell is not burnable."""
        cell = self.cells[row][col]
        if not cell.is_burnable:
            return False
        cell.fire_state = FireState.BURNING
        cell.burn_time  = timestep
        return True

    def reset(self) -> None:
        """Reset all fire states to UNBURNED."""
        for row in self.cells:
            for cell in row:
                cell.fire_state = FireState.UNBURNED
                cell.retardant  = 0
                cell.burn_time  = -1

    def total_value(self, state: FireState = FireState.UNBURNED) -> float:
        """Sum of cell values in the given fire state."""
        return sum(
            c.value
            for row in self.cells
            for c in row
            if c.fire_state == state
        )

    def fire_state_array(self) -> np.ndarray:
        """Return (rows, cols) int array of fire states for visualization."""
        return np.array(
            [[c.fire_state.value for c in row] for row in self.cells]
        )

    def rgb_array(self) -> np.ndarray:
        """Return (rows, cols, 3) uint8 RGB array for imshow."""
        import re
        rgb = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                hex_c = cell.color.lstrip("#")
                rgb[r, c] = [int(hex_c[i:i+2], 16) for i in (0, 2, 4)]
        return rgb


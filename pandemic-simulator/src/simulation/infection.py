"""
infection.py — Stochastic SIR simulator on a city network.

Model summary
─────────────
Each time step represents roughly one week of real-world time.

Within each city the disease spreads according to a discrete-time
SIR (Susceptible → Infected → Recovered) process:

    new_infections ~ Binomial(S, β · I/N)
    new_recoveries ~ Binomial(I, γ)

Between cities, infected travellers seed new cases proportional to
the edge travel-weight and the source city's infection rate.

Player roles act as modifiers on β and γ, or trigger one-off actions
(quarantine, mass treatment, etc.).

Outcomes
────────
• WIN  — cure_progress reaches 1.0 before the pandemic spirals.
• LOSE — more than 50 % of global population is simultaneously infected.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Any

from .network import City
from .roles import PlayerRole


@dataclass
class InfectionParams:
    """Tuneable knobs for the simulation engine."""
    base_spread_rate: float   = 0.45   # β: within-city transmission per step
    recovery_rate: float      = 0.04   # γ: recovery probability per infected per step
    travel_spread_factor: float = 0.18 # extra seeding along travel routes
    cure_gain_per_step: float = 0.013  # baseline cure-research progress each step (~77 steps)
    outbreak_threshold: int   = 500    # infected count that triggers an "outbreak" event
    loss_threshold: float     = 0.40   # fraction of global pop infected → loss condition


class InfectionSimulator:
    """
    Drives a stochastic SIR pandemic across a weighted city network.

    Parameters
    ----------
    graph  : NetworkX graph with 'weight' edge attributes
    cities : dict mapping city name → City instance
    params : InfectionParams (uses defaults if omitted)
    seed   : random seed for reproducibility
    """

    def __init__(
        self,
        graph: nx.Graph,
        cities: dict[str, City],
        params: InfectionParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.graph  = graph
        self.cities = cities
        self.params = params or InfectionParams()
        self.rng    = np.random.default_rng(seed)

        self.step_count: int   = 0
        self.cure_progress: float = 0.0
        self.outbreaks: list[tuple[int, str]] = []   # (step, city_name)
        self.history: list[dict[str, Any]] = []
        self._total_infected_ever: int = 0

    # ------------------------------------------------------------------ #
    #  Setup                                                               #
    # ------------------------------------------------------------------ #

    def seed_infection(self, city_name: str, count: int = 3) -> None:
        """Place initial infections in a city."""
        city = self.cities[city_name]
        count = min(count, city.susceptible)
        city.infected += count
        self._total_infected_ever += count

    # ------------------------------------------------------------------ #
    #  Core step                                                           #
    # ------------------------------------------------------------------ #

    def step(self, roles: list[PlayerRole] | None = None) -> bool:
        """
        Advance the simulation by one time step.

        Returns True when the simulation has reached a terminal state
        (win or loss), False otherwise.
        """
        roles = roles or []
        new_infections: dict[str, int] = {}
        new_recoveries: dict[str, int] = {}

        # ── Within-city spread ──────────────────────────────────────────
        for name, city in self.cities.items():
            if city.quarantined or city.infected == 0:
                continue

            spread_rate = self.params.base_spread_rate
            for role in roles:
                spread_rate = role.modify_spread_rate(name, spread_rate)

            # New infections
            p_infect = min(spread_rate * city.infection_rate, 1.0)
            new_inf  = int(self.rng.binomial(city.susceptible, p_infect))
            new_infections[name] = new_infections.get(name, 0) + new_inf

            # Outbreak detection
            if city.infected + new_inf >= self.params.outbreak_threshold:
                self.outbreaks.append((self.step_count, name))

            # Recoveries
            new_rec = int(self.rng.binomial(city.infected, self.params.recovery_rate))
            new_recoveries[name] = new_rec

        # ── Travel-based spread ─────────────────────────────────────────
        for u, v, data in self.graph.edges(data=True):
            weight = data.get("weight", 0.5)
            for src_name, dst_name in [(u, v), (v, u)]:
                src = self.cities[src_name]
                dst = self.cities[dst_name]
                if src.infected == 0 or dst.quarantined or src.quarantined:
                    continue
                # Small cohort of travellers; proportion that are infectious
                travel_pool = max(1, dst.susceptible // 200)
                p_travel = min(
                    self.params.travel_spread_factor * weight * src.infection_rate,
                    1.0,
                )
                arrivals = int(self.rng.binomial(travel_pool, p_travel))
                new_infections[dst_name] = new_infections.get(dst_name, 0) + arrivals

        # ── Apply updates ───────────────────────────────────────────────
        for name, count in new_infections.items():
            city = self.cities[name]
            actual = min(count, city.susceptible)
            city.infected += actual
            self._total_infected_ever += actual

        for name, count in new_recoveries.items():
            city = self.cities[name]
            city.infected  = max(0, city.infected - count)
            city.recovered += count

        # ── Cure research ───────────────────────────────────────────────
        gain = self.params.cure_gain_per_step
        for role in roles:
            gain = role.modify_cure_rate(gain)
        self.cure_progress = min(1.0, self.cure_progress + gain)

        # ── Record history ──────────────────────────────────────────────
        self._record_state()
        self.step_count += 1

        return self.is_terminal()

    # ------------------------------------------------------------------ #
    #  Terminal conditions                                                 #
    # ------------------------------------------------------------------ #

    def is_cured(self) -> bool:
        return self.cure_progress >= 1.0

    def is_lost(self) -> bool:
        total_inf = sum(c.infected for c in self.cities.values())
        total_pop = sum(c.population for c in self.cities.values())
        return total_inf / total_pop >= self.params.loss_threshold

    def is_terminal(self) -> bool:
        return self.is_cured() or self.is_lost()

    # ------------------------------------------------------------------ #
    #  Convenience properties                                              #
    # ------------------------------------------------------------------ #

    @property
    def total_currently_infected(self) -> int:
        return sum(c.infected for c in self.cities.values())

    @property
    def global_infection_rate(self) -> float:
        total_pop = sum(c.population for c in self.cities.values())
        return self.total_currently_infected / total_pop if total_pop else 0.0

    # ------------------------------------------------------------------ #
    #  History                                                             #
    # ------------------------------------------------------------------ #

    def _record_state(self) -> None:
        record: dict[str, Any] = {
            "step":           self.step_count,
            "cure_progress":  round(self.cure_progress, 4),
            "total_infected": self.total_currently_infected,
            "outbreaks_total": len(self.outbreaks),
        }
        for name, city in self.cities.items():
            record[f"{name}_infected"]  = city.infected
            record[f"{name}_recovered"] = city.recovered
        self.history.append(record)

    def summary(self) -> dict[str, Any]:
        """High-level outcome summary."""
        return {
            "outcome":              "WIN" if self.is_cured() else "LOSS",
            "steps":                self.step_count,
            "cure_progress":        round(self.cure_progress, 4),
            "total_ever_infected":  self._total_infected_ever,
            "outbreaks":            len(self.outbreaks),
            "worst_city":           max(
                self.cities, key=lambda n: self.cities[n].recovered + self.cities[n].infected
            ),
        }

"""
roles.py — Player role definitions.

Each role mirrors a Pandemic-style character ability, expressed as
modifier methods that the simulator calls each step. Roles can:
  - reduce local or global spread rates
  - accelerate cure research
  - take special one-off actions (quarantine, mass treatment, etc.)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .infection import InfectionSimulator


class PlayerRole:
    """Abstract base for all player roles."""
    name: str = "None"
    description: str = ""

    def modify_spread_rate(self, city_name: str, rate: float) -> float:
        """Return an adjusted local spread rate for *city_name*."""
        return rate

    def modify_cure_rate(self, rate: float) -> float:
        """Return an adjusted cure-progress gain per step."""
        return rate

    def special_action(self, sim: "InfectionSimulator", city_name: str) -> str:
        """
        Execute a one-time special action.
        Returns a human-readable description of what happened.
        """
        return f"{self.name} has no special action."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Medic(PlayerRole):
    """
    Stationed in one city; drastically reduces local spread and can
    mass-treat infected citizens once per call to special_action().
    """
    name = "Medic"
    description = (
        "Reduces spread rate by 60 % in their stationed city. "
        "Special action: immediately recover up to 200 infected citizens."
    )
    SPREAD_MULTIPLIER = 0.40
    TREATMENT_CAPACITY = 200

    def __init__(self, stationed_city: str) -> None:
        self.stationed_city = stationed_city

    def modify_spread_rate(self, city_name: str, rate: float) -> float:
        if city_name == self.stationed_city:
            return rate * self.SPREAD_MULTIPLIER
        return rate

    def special_action(self, sim: "InfectionSimulator", city_name: str) -> str:
        city = sim.cities.get(city_name)
        if not city or city.infected == 0:
            return f"Medic: no infected in {city_name}."
        treated = min(city.infected, self.TREATMENT_CAPACITY)
        city.infected -= treated
        city.recovered += treated
        return f"Medic treated {treated:,} citizens in {city_name}."


class Scientist(PlayerRole):
    """Dramatically accelerates cure research."""
    name = "Scientist"
    description = "Cure research progresses 2.5× faster."
    CURE_MULTIPLIER = 2.5

    def modify_cure_rate(self, rate: float) -> float:
        return rate * self.CURE_MULTIPLIER


class QuarantineSpecialist(PlayerRole):
    """Can lock down a city, stopping all inbound travel spread."""
    name = "Quarantine Specialist"
    description = (
        "Special action: quarantine a city, blocking all travel-based spread "
        "into and out of it."
    )

    def special_action(self, sim: "InfectionSimulator", city_name: str) -> str:
        city = sim.cities.get(city_name)
        if not city:
            return f"Quarantine Specialist: {city_name!r} not found."
        city.quarantined = True
        return f"Quarantine Specialist locked down {city_name}."


class Epidemiologist(PlayerRole):
    """Provides a passive global spread reduction through coordinated surveillance."""
    name = "Epidemiologist"
    description = "Reduces spread rate by 25 % in every city globally."
    SPREAD_MULTIPLIER = 0.75

    def modify_spread_rate(self, city_name: str, rate: float) -> float:
        return rate * self.SPREAD_MULTIPLIER


class Dispatcher(PlayerRole):
    """
    Coordinates travel restrictions, slowing the cross-city cascade.
    Provides a moderate spread reduction everywhere.
    """
    name = "Dispatcher"
    description = "Reduces spread rate by 15 % globally via travel coordination."
    SPREAD_MULTIPLIER = 0.85

    def modify_spread_rate(self, city_name: str, rate: float) -> float:
        return rate * self.SPREAD_MULTIPLIER


# Convenience registry for Monte Carlo experiments
ALL_ROLES: dict[str, type[PlayerRole]] = {
    "medic":                   Medic,
    "scientist":               Scientist,
    "quarantine_specialist":   QuarantineSpecialist,
    "epidemiologist":          Epidemiologist,
    "dispatcher":              Dispatcher,
}

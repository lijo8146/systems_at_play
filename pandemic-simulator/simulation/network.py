"""
network.py — City nodes and global connection graph.

Each city tracks its own SIR compartments. The graph edges carry
travel-weight values representing how heavily two cities are connected
(e.g. flight volume, shared border traffic).
"""

from __future__ import annotations
import networkx as nx
from dataclasses import dataclass, field


@dataclass
class City:
    name: str
    population: int
    infected: int = 0
    recovered: int = 0
    quarantined: bool = False

    @property
    def susceptible(self) -> int:
        return max(0, self.population - self.infected - self.recovered)

    @property
    def infection_rate(self) -> float:
        """Fraction of the city that is currently infectious."""
        return self.infected / self.population if self.population else 0.0

    def __repr__(self) -> str:
        return (
            f"City({self.name!r}, pop={self.population:,}, "
            f"inf={self.infected:,}, rec={self.recovered:,})"
        )


def build_world_network() -> tuple[nx.Graph, dict[str, City]]:
    """
    Return a NetworkX graph and city dictionary for a simplified world network.

    Edge weights represent relative travel volume (0–1 scale).
    Higher weight → more passenger movement → faster cross-city spread.
    """
    cities: dict[str, City] = {
        "Atlanta":    City("Atlanta",    500_000),
        "New York":   City("New York",   8_000_000),
        "London":     City("London",     9_000_000),
        "Paris":      City("Paris",      2_200_000),
        "Tokyo":      City("Tokyo",     14_000_000),
        "Beijing":    City("Beijing",   21_000_000),
        "Lagos":      City("Lagos",     15_000_000),
        "Cairo":      City("Cairo",     10_000_000),
        "Sydney":     City("Sydney",     5_300_000),
        "São Paulo":  City("São Paulo", 12_000_000),
    }

    G = nx.Graph()
    G.add_nodes_from(cities.keys())

    # (city_a, city_b, travel_weight)
    edges = [
        ("Atlanta",   "New York",  0.80),
        ("New York",  "London",    0.90),
        ("New York",  "São Paulo", 0.60),
        ("London",    "Paris",     0.95),
        ("London",    "Lagos",     0.50),
        ("London",    "Cairo",     0.60),
        ("Paris",     "Cairo",     0.45),
        ("Cairo",     "Lagos",     0.40),
        ("Beijing",   "Tokyo",     0.85),
        ("Beijing",   "London",    0.70),
        ("Tokyo",     "Sydney",    0.60),
        ("Atlanta",   "Lagos",     0.30),
        ("São Paulo", "Lagos",     0.40),
        ("Sydney",    "Beijing",   0.50),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G, cities

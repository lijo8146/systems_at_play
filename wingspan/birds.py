"""
birds.py bird archetypes and power types for the Wingspan engine model.

We model the forest habitat (gain food action) because it produces the
clearest engine analysis: birds with "when activated" powers that generate
food, tuck cards, or cache food create measurable compound output over time.

Power types modeled
NONE          : No activation power. Bird contributes points but no engine value.
GAIN_FOOD     : Gain N food from supply when this bird is activated.
TUCK_FROM_DECK: Tuck N cards from the top of the deck under this bird
                (counts toward end-game points but also primes the deck cycle).
CACHE_FOOD    : Cache N food on this bird (protected from robins, counts at end).
GAIN_CARDS    : Draw N cards when activated.
COPY_NEIGHBOR : Activate the power of one adjacent bird in the same habitat.
ROLL_FOOD     : Roll N food dice, gain matching food types (stochastic).
REPEAT_ACTION : Activate all birds in this habitat again (chain trigger).

Bird cost
Each bird has a food_cost (eggs needed to play are omitted for simplicity so
we focus on the food economy). The engine efficiency metric is:
    efficiency = expected_output_per_activation/food_cost_to_play

This captures the core strategic tension: a cheap bird that generates
moderate output may be better than an expensive bird with high output,
depending on how many turns remain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class PowerType(Enum):
    NONE          = auto()
    GAIN_FOOD     = auto()
    TUCK_FROM_DECK= auto()
    CACHE_FOOD    = auto()
    GAIN_CARDS    = auto()
    COPY_NEIGHBOR = auto()
    ROLL_FOOD     = auto()
    REPEAT_ACTION = auto()


@dataclass
class Bird:
    """
    A Wingspan bird card archetype.

    Attributes
    name         : Display name (flavour + archetype role)
    habitat      : "forest", "grassland", or "wetland"
    food_cost    : Total food required to play this bird
    points       : Victory points printed on the card
    power_type   : Category of when-activated power
    power_value  : Numeric parameter for the power (e.g. N in "gain N food")
    nest_type    : "platform", "bowl", "cavity", "ground", "star"
    wingspan_cm  : Bird's wingspan in cm (flavour; used as a tiebreaker)
    description  : Human-readable power description
    """
    name        : str
    habitat     : str
    food_cost   : int
    points      : int
    power_type  : PowerType
    power_value : int
    nest_type   : str       = "platform"
    wingspan_cm : int       = 50
    description : str       = ""

    @property
    def expected_food_output(self) -> float:
        """
        Expected food-equivalent units produced per single activation.

        Cards drawn are valued at 0.5 food-equivalent (rough conversion:
        cards enable future plays which require food, so they have
        indirect food value).
        Tucked cards score end-game points valued at 0.3 food-equivalent.
        Cached food is treated as 1.0 food-equivalent (it's food, just stored).
        REPEAT_ACTION is valued as the mean output of a typical forest bird.
        COPY_NEIGHBOR is context-dependent; estimated at 0.8 food-equivalent.
        """
        if self.power_type == PowerType.GAIN_FOOD:
            return float(self.power_value)
        elif self.power_type == PowerType.ROLL_FOOD:
            # Each die has 6 faces; roughly 1/2 will match (3 food types and 6 faces)
            # Actually Wingspan dice have: worm, berry, seed, fish, rat, rat
            # Expected value per die ≈ 5/6 (5 of 6 faces produce food)
            return self.power_value * (5 / 6)
        elif self.power_type == PowerType.CACHE_FOOD:
            return float(self.power_value)   # cached food = food
        elif self.power_type == PowerType.GAIN_CARDS:
            return self.power_value * 0.5    # card ≈ 0.5 food-equivalent
        elif self.power_type == PowerType.TUCK_FROM_DECK:
            return self.power_value * 0.3    # tuck ≈ 0.3 food-equivalent
        elif self.power_type == PowerType.COPY_NEIGHBOR:
            return 0.8                        # estimated average neighbor output
        elif self.power_type == PowerType.REPEAT_ACTION:
            return 1.2                        # triggers all birds again ~= avg output
        else:
            return 0.0                        # NONE has no engine contribution

    @property
    def efficiency(self) -> float:
        """Expected food-equivalent output per food spent to play."""
        if self.food_cost == 0:
            return float("inf")
        return self.expected_food_output / self.food_cost

    def __repr__(self) -> str:
        return (f"Bird({self.name!r}, {self.points}pts, "
                f"cost={self.food_cost}, {self.power_type.name}, "
                f"val={self.power_value})")


# Forest habitat archetypes
# These represent the ~8 power archetypes found in the forest habitat.
# Each archetype appears in multiple cards in the real game; these are
# representative examples at different cost/points tiers.

FOREST_BIRDS: list[Bird] = [

    # NONE: pure points, no engine
    Bird("American Kestrel",  "forest", food_cost=1, points=3,
         power_type=PowerType.NONE, power_value=0,
         nest_type="cavity", wingspan_cm=55,
         description="No activation power. High points per food spent."),

    Bird("Bald Eagle",        "forest", food_cost=4, points=9,
         power_type=PowerType.NONE, power_value=0,
         nest_type="platform", wingspan_cm=213,
         description="High points, no engine contribution."),

    # GAIN_FOOD: direct food generation
    Bird("Common Raven",      "forest", food_cost=2, points=4,
         power_type=PowerType.GAIN_FOOD, power_value=1,
         nest_type="platform", wingspan_cm=130,
         description="When activated: gain 1 food from supply."),

    Bird("Steller's Jay",     "forest", food_cost=2, points=5,
         power_type=PowerType.GAIN_FOOD, power_value=1,
         nest_type="platform", wingspan_cm=43,
         description="When activated: gain 1 food from supply."),

    Bird("Clark's Nutcracker","forest", food_cost=3, points=6,
         power_type=PowerType.GAIN_FOOD, power_value=2,
         nest_type="platform", wingspan_cm=61,
         description="When activated: gain 2 food from supply."),

    Bird("Great Horned Owl",  "forest", food_cost=3, points=7,
         power_type=PowerType.GAIN_FOOD, power_value=2,
         nest_type="platform", wingspan_cm=137,
         description="When activated: gain 2 food from supply."),

    Bird("Northern Goshawk",  "forest", food_cost=4, points=7,
         power_type=PowerType.GAIN_FOOD, power_value=3,
         nest_type="platform", wingspan_cm=105,
         description="When activated: gain 3 food from supply."),

    # ROLL_FOOD: stochastic food via dice
    Bird("Wild Turkey",       "forest", food_cost=2, points=4,
         power_type=PowerType.ROLL_FOOD, power_value=1,
         nest_type="ground", wingspan_cm=150,
         description="When activated: roll 1 food die, gain matching food."),

    Bird("Ruffed Grouse",     "forest", food_cost=2, points=5,
         power_type=PowerType.ROLL_FOOD, power_value=2,
         nest_type="ground", wingspan_cm=56,
         description="When activated: roll 2 food dice, gain matching food."),

    Bird("Blue Grouse",       "forest", food_cost=3, points=6,
         power_type=PowerType.ROLL_FOOD, power_value=3,
         nest_type="ground", wingspan_cm=65,
         description="When activated: roll 3 food dice, gain matching food."),

    # CACHE_FOOD: store food on card
    Bird("Black-capped Chickadee","forest", food_cost=1, points=3,
         power_type=PowerType.CACHE_FOOD, power_value=1,
         nest_type="cavity", wingspan_cm=21,
         description="When activated: cache 1 food on this card."),

    Bird("Red-breasted Nuthatch","forest", food_cost=2, points=4,
         power_type=PowerType.CACHE_FOOD, power_value=2,
         nest_type="cavity", wingspan_cm=23,
         description="When activated: cache 2 food on this card."),

    # TUCK_FROM_DECK: tuck cards for end-game points
    Bird("Downy Woodpecker",  "forest", food_cost=1, points=2,
         power_type=PowerType.TUCK_FROM_DECK, power_value=1,
         nest_type="cavity", wingspan_cm=30,
         description="When activated: tuck 1 card from deck under this bird."),

    Bird("Pileated Woodpecker","forest", food_cost=3, points=6,
         power_type=PowerType.TUCK_FROM_DECK, power_value=3,
         nest_type="cavity", wingspan_cm=73,
         description="When activated: tuck 3 cards from deck under this bird."),

    # GAIN_CARDS: draw cards for hand
    Bird("Blue Jay",          "forest", food_cost=2, points=4,
         power_type=PowerType.GAIN_CARDS, power_value=1,
         nest_type="platform", wingspan_cm=43,
         description="When activated: draw 1 card from the deck or bird tray."),

    Bird("Gray Jay",          "forest", food_cost=2, points=5,
         power_type=PowerType.GAIN_CARDS, power_value=2,
         nest_type="platform", wingspan_cm=37,
         description="When activated: draw 2 cards."),

    # COPY_NEIGHBOR: activate an adjacent bird's power
    Bird("European Starling", "forest", food_cost=2, points=4,
         power_type=PowerType.COPY_NEIGHBOR, power_value=1,
         nest_type="cavity", wingspan_cm=37,
         description="When activated: activate the power of one adjacent bird."),

    Bird("Brown-headed Cowbird","forest",food_cost=1, points=2,
         power_type=PowerType.COPY_NEIGHBOR, power_value=1,
         nest_type="ground", wingspan_cm=36,
         description="When activated: copy one adjacent bird's power."),

    # REPEAT_ACTION: activate all birds again
    Bird("Purple Martin",     "forest", food_cost=3, points=5,
         power_type=PowerType.REPEAT_ACTION, power_value=1,
         nest_type="cavity", wingspan_cm=43,
         description="When activated: activate all birds in this habitat again."),

]

# Index by name for easy lookup
BIRD_BY_NAME: dict[str, Bird] = {b.name: b for b in FOREST_BIRDS}


def top_birds_by_efficiency(n: int = 10) -> list[Bird]:
    """Return the n most efficient birds (output per food cost)."""
    return sorted(FOREST_BIRDS, key=lambda b: b.efficiency, reverse=True)[:n]


def birds_by_power_type(power_type: PowerType) -> list[Bird]:
    """Return all birds with the given power type."""
    return [b for b in FOREST_BIRDS if b.power_type == power_type]

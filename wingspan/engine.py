"""
engine.py wingspan forest engine simulation and analysis.

The forest engine
When you take the "gain food" action in the forest habitat, you gain
food from the birdfeeder AND trigger the activation power of every bird
in your forest, from right to left.

Engine output per action = base food gained + sum of all bird powers triggered.

As you add birds to your forest, each new bird adds its power to every
future activation. The key insight: birds with ongoing production powers
(GAIN_FOOD, ROLL_FOOD, CACHE_FOOD) compound across the entire game.

Turn structure (simplified)
A Wingspan game has 4 rounds of decreasing length:
  Round 1: 8 turns each player
  Round 2: 7 turns
  Round 3: 6 turns
  Round 4: 5 turns
Total: 26 turns per player

On each turn a player either:
  - Plays a bird (spends food/eggs, adds bird to habitat)
  - Activates a habitat (gains resources and triggers all birds there)
  - Takes another action (draw cards, lay eggs)

This simulation focuses on the forest habitat engine. We model a player
who builds their forest progressively and tracks total food production
over the game.

Metrics
engine_output_per_activation : total food-equivalent units generated
                                per "gain food" action
cumulative_output             : running total of food-equivalent units
                                across all activations
food_efficiency               : cumulative_output/total_food_spent_on_birds
breakeven_turn                : when cumulative output exceeds cost to play
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import pandas as pd

from birds import Bird, PowerType, FOREST_BIRDS


# Game constants

ROUND_LENGTHS = [8, 7, 6, 5]   # turns per round
TOTAL_TURNS   = sum(ROUND_LENGTHS)   # 26 turns
BASE_FOOD_PER_ACTION = 1             # food from birdfeeder (simplified: 1/action)


# Engine state

@dataclass
class EngineState:
    """
    The current state of a player's forest engine.

    forest       : ordered list of birds in the forest (right to left activation)
    food_reserve : food available for playing birds
    turn         : current turn (0-indexed)
    round        : current round (0-indexed)
    history      : list of (turn, event, value) records for analysis
    """
    forest      : list[Bird]  = field(default_factory=list)
    food_reserve: float       = 0.0
    total_food_spent: float   = 0.0
    turn        : int         = 0
    round_idx   : int         = 0
    history     : list[dict]  = field(default_factory=list)

    @property
    def engine_size(self) -> int:
        return len(self.forest)

    @property
    def base_output(self) -> float:
        return BASE_FOOD_PER_ACTION

    def activation_output(self, rng: np.random.Generator | None = None) -> float:
        """
        Compute total food-equivalent output for one "gain food" activation.
        Stochastic for ROLL_FOOD birds.
        """
        total = self.base_output
        for bird in self.forest:
            if bird.power_type == PowerType.ROLL_FOOD:
                if rng is not None:
                    # Each die: 5/6 chance of producing food
                    rolls = rng.binomial(bird.power_value, 5/6)
                    total += float(rolls)
                else:
                    total += bird.expected_food_output
            elif bird.power_type == PowerType.REPEAT_ACTION:
                # Activates all other birds once more
                for other in self.forest:
                    if other is not bird:
                        total += other.expected_food_output
            else:
                total += bird.expected_food_output
        return total


def play_bird(state: EngineState, bird: Bird) -> bool:
    """
    Attempt to play a bird into the forest.
    Returns True if successful (enough food), False otherwise.
    Deducts food cost and adds bird to forest.
    """
    if state.food_reserve < bird.food_cost:
        return False
    state.food_reserve -= bird.food_cost
    state.total_food_spent += bird.food_cost
    state.forest.append(bird)
    state.history.append({
        "turn":   state.turn,
        "event":  "play_bird",
        "bird":   bird.name,
        "cost":   bird.food_cost,
        "points": bird.points,
        "engine_size": state.engine_size,
    })
    return True


# Simulation

@dataclass
class SimConfig:
    """Configuration for an engine simulation run."""
    build_sequence  : list[Bird]   # birds to play, in order of priority
    n_games         : int  = 500
    seed            : int  = 42
    play_threshold  : int  = 2     # min turns between bird plays
    starting_food   : int  = 5     # food available at game start


def simulate_engine(config: SimConfig) -> pd.DataFrame:
    """
    Simulate N games of forest engine building.

    Strategy: on each turn, either play the next bird in the build_sequence
    (if we have enough food) or take the "gain food" action.

    Returns a DataFrame with one row per (game, turn):
        game, turn, round, action, engine_size,
        activation_output, cumulative_output,
        food_reserve, total_food_spent
    """
    rng  = np.random.default_rng(config.seed)
    rows = []

    for game in range(config.n_games):
        state = EngineState(food_reserve=float(config.starting_food))
        build_queue = list(config.build_sequence)  # copy mutable
        cumulative  = 0.0
        turn_global = 0

        for round_idx, round_len in enumerate(ROUND_LENGTHS):
            state.round_idx = round_idx
            for _ in range(round_len):
                state.turn = turn_global

                # Decision: play a bird or gain food?
                action = "gain_food"
                bird_played = None

                if build_queue:
                    next_bird = build_queue[0]
                    if state.food_reserve >= next_bird.food_cost:
                        play_bird(state, next_bird)
                        build_queue.pop(0)
                        action = "play_bird"
                        bird_played = next_bird.name

                if action == "gain_food":
                    output = state.activation_output(rng=rng)
                    state.food_reserve += output
                    cumulative += output
                else:
                    output = 0.0

                rows.append({
                    "game":              game,
                    "turn":              turn_global,
                    "round":             round_idx + 1,
                    "action":            action,
                    "bird_played":       bird_played,
                    "engine_size":       state.engine_size,
                    "activation_output": output,
                    "cumulative_output": cumulative,
                    "food_reserve":      state.food_reserve,
                    "total_food_spent":  state.total_food_spent,
                })
                turn_global += 1

    return pd.DataFrame(rows)


# Single-bird marginal value analysis

def marginal_value(
    bird: Bird,
    existing_forest: list[Bird],
    turns_remaining: int,
    n_activations_per_turn: float = 0.5,
) -> float:
    """
    Expected additional food-equivalent output from adding one bird to
    an existing forest, across the turns remaining.

    n_activations_per_turn: fraction of turns spent on "gain food" action.
    Typical value ~0.5 (half your turns build birds, half gain food).

    Returns: total expected output gain - food cost to play.
    """
    activations = turns_remaining * n_activations_per_turn
    gain = bird.expected_food_output * activations
    return gain - bird.food_cost


def breakeven_turn(bird: Bird, n_activations_per_turn: float = 0.5) -> float:
    """
    Turn at which a bird's cumulative output equals its food cost.
    Turns before this: net negative (still in debt). After: net positive.
    """
    if bird.expected_food_output == 0:
        return float("inf")
    return bird.food_cost / (bird.expected_food_output * n_activations_per_turn)


# Preset build sequences for comparison

def make_config_pure_points() -> SimConfig:
    """Plays high-point, no-power birds. Baseline: no engine."""
    from birds import BIRD_BY_NAME
    sequence = [
        BIRD_BY_NAME["American Kestrel"],
        BIRD_BY_NAME["Bald Eagle"],
        BIRD_BY_NAME["American Kestrel"],
    ]
    return SimConfig(build_sequence=sequence, n_games=500,
                     play_threshold=2, starting_food=4)


def make_config_food_engine() -> SimConfig:
    """Builds a food-generating engine (GAIN_FOOD birds)."""
    from birds import BIRD_BY_NAME
    sequence = [
        BIRD_BY_NAME["Common Raven"],
        BIRD_BY_NAME["Clark's Nutcracker"],
        BIRD_BY_NAME["Great Horned Owl"],
        BIRD_BY_NAME["Northern Goshawk"],
    ]
    return SimConfig(build_sequence=sequence, n_games=500,
                     play_threshold=2, starting_food=4)


def make_config_dice_engine() -> SimConfig:
    """Builds a stochastic dice-rolling food engine (ROLL_FOOD birds)."""
    from birds import BIRD_BY_NAME
    sequence = [
        BIRD_BY_NAME["Wild Turkey"],
        BIRD_BY_NAME["Ruffed Grouse"],
        BIRD_BY_NAME["Blue Grouse"],
        BIRD_BY_NAME["Ruffed Grouse"],
    ]
    return SimConfig(build_sequence=sequence, n_games=500,
                     play_threshold=2, starting_food=4)


def make_config_mixed() -> SimConfig:
    """Mixed engine: food generation + card draw + caching."""
    from birds import BIRD_BY_NAME
    sequence = [
        BIRD_BY_NAME["Black-capped Chickadee"],
        BIRD_BY_NAME["Common Raven"],
        BIRD_BY_NAME["Blue Jay"],
        BIRD_BY_NAME["Clark's Nutcracker"],
        BIRD_BY_NAME["Red-breasted Nuthatch"],
    ]
    return SimConfig(build_sequence=sequence, n_games=500,
                     play_threshold=2, starting_food=4)

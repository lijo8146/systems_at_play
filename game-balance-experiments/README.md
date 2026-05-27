# Game Balance Experiments

> *A game is balanced when no single strategy dominates, and finding the ones that do is half the fun.*

## The question

How do you know if a game is fair? One player's intuition says "the blue faction is overpowered." Another says "you just don't know how to counter it." Who's right?

Simulation answers this without argument: run the game thousands of times under different strategies and measure the outcomes.

This module experiments with game balance mechanics — what happens to win rates, variance, and dominant strategies when you tweak the rules?

## Experiments

### 1. Dice duel (`dice_duel.py`)

Two players each choose a dice configuration. They roll simultaneously; higher roll wins. Which dice configurations are dominant? Are there non-transitive dice (where A beats B, B beats C, but C beats A)?

Concepts: **dominant strategies, non-transitive relations, expected value vs variance**

### 2. Resource race (`resource_race.py`)

Players accumulate resources each turn and spend them to buy actions. How does changing the cost/benefit ratio of an action affect whether it becomes the obvious dominant choice? At what point does a strategy stop being a "choice" and become a requirement?

Concepts: **payoff matrices, dominant strategies, Nash equilibrium**

### 3. Snowball analysis (`snowball.py`)

Many games have mechanics that reward the leader ("snowballing"). How much of a lead advantage is needed to make the outcome inevitable? How do catch-up mechanics change this?

Concepts: **feedback loops, runaway leader, variance vs skill**

## Run it
```bash
pip install -r requirements.txt
python dice_duel.py        
python resource_race.py    
```

## What to try
- Design your own dice configurations and test whether they create balanced matchups
- Model a real board game mechanic (ex. Catan's development cards) and check for dominant strategies
- Add a "catch-up" mechanic to `snowball.py` and measure how much it reduces outcome predictability

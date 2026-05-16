# Catan Probability Models

> *Where should you put your first settlement? The answer is a probability distribution.*

## The question

In *Settlers of Catan*, every resource hex has a number (2–12). On each turn, two dice are rolled. If the total matches a number on a hex adjacent to your settlement, you get a resource card.

The question is: **given a specific board layout, which settlement positions have the highest expected resource yield?**

This turns out to be a beautifully clean probability problem.

## The math

Two six-sided dice produce outcomes with known frequencies:

| Roll | Ways to roll | Probability | Pip dots |
|------|-------------|-------------|----------|
| 2    | 1           | 2.78 %      | •        |
| 3    | 2           | 5.56 %      | ••       |
| 4    | 3           | 8.33 %      | •••      |
| 5    | 4           | 11.11 %     | ••••     |
| 6    | 5           | 13.89 %     | •••••    |
| 7    | 6           | 16.67 %     | (robber) |
| 8    | 5           | 13.89 %     | •••••    |
| 9    | 4           | 11.11 %     | ••••     |
| 10   | 3           | 8.33 %      | •••      |
| 11   | 2           | 5.56 %      | ••       |
| 12   | 1           | 2.78 %      | •        |

The physical Catan board represents this with **pip dots** so that the 6 and 8 tiles have 5 dots each, 5 and 9 have 4, etc. Your expected resources per turn from a settlement equals the sum of pip values of adjacent hexes divided by 36.

A settlement on three 6/8/9 hexes would have expected yield ≈ (5+5+4)/36 ≈ **0.39 resources/turn**.

## What this module models
- **`dice.py`** : probability distributions, expected values, pip calculations
- **`board.py`** : standard Catan board layout with settlement vertex analysis
- **`simulator.py`** : Monte Carlo simulation of full games (resource production, robber impact)
- **`plots.py`** : heatmaps, production histograms, settlement comparison charts
- **`run_analysis.py`** : entry point that runs all analyses and saves plots

## Run it
```bash
pip install -r requirements.txt
python run_analysis.py
```

Outputs land in `output/`:
- `pip_distribution.png` : the probability curve for 2d6
- `settlement_ev_heatmap.png` : board heatmap coloured by settlement EV
- `resource_production.png` : simulated resource yield distributions by position
- `robber_impact.png` : how much the robber on a high-value hex costs you

## What to try
- Change the board layout in `board.py` and see how dramatically EV shifts
- Modify `simulator.py` to track *diversity* of resources, not just volume
- Add port locations (3:1 and 2:1 ports change which settlements are strategically best)
- Model a "balanced" board vs a "chaotic" one by adjusting number placements

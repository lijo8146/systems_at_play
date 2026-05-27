# Systems at Play 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20418769.svg)](https://doi.org/10.5281/zenodo.20418769)

> *Complex systems become intuitive when you can play with them.*

**Systems at Play** is a curated laboratory of computational experiments using games, puzzles, and simulations as lenses for exploring probability, strategy, emergence, and decision-making.

Each experiment is self-contained and designed to be readable by someone new to the topic. The goal is both analytical and educational: making rigorous ideas feel approachable by grounding them in systems people already find interesting.

## Experiments
| Project | What it explores | Key ideas |
|---|---|---|
| [ Pandemic Simulator](./pandemic-simulator/) | Disease spread on a city network | SIR model, Monte Carlo, graph dynamics |
| [ Catan Probability Models](./catan-probability-models/) | Settlement placement & resource yields | Expected value, probability distributions, spatial optimization |
| [ Puzzle Solvers](./puzzle-solvers/) | Sudoku and maze generation/solving | Backtracking, constraint propagation, BFS/DFS |
| [ Game Balance Experiments](./game-balance-experiments/) | How small rule changes break (or fix) games | Dominant strategies, variance, payoff matrices |
| [ Decision Tree Visualizations](./decision-tree-visualizations/) | Search and strategy in two-player games | Minimax, game trees, evaluation functions |
| [ Logic Grid Solver ](./logic_grid_solver) | A logic grid puzzle gives you N categories with N items each and a set of clues | Constraint Satisfaction Problem (CSP)  |
| [ Wingspan ](./wingspan) | Stochastic Modeling of Resource Production | Breakeven analysis, engine output over time, variance, marginal value heatmap |

## Philosophy
A lot of computational thinking is taught through problems that feel abstract and arbitrary. This repository takes a different approach:

**Games and simulations are already complex systems.** A Catan board is a probability distribution problem. A maze is a graph traversal problem. A pandemic is a network dynamics problem. Using them as entry points makes the underlying ideas easier to engage with, remember, and build on.

Each experiment tries to answer a question in the form: *"What actually happens if…?"*

- What actually happens to your Catan odds if the robber sits on your 6 all game?
- What happens to disease spread when you add a single quarantine specialist?
- What happens to game balance when you change one rule?

## Who this is for
- **Students** learning probability, algorithms, or simulation for the first time
- **Educators** looking for worked examples that connect math to familiar contexts
- **Developers** building a portfolio that shows systems thinking
- **Curious people** who want to poke at how things work

Every module includes:
- A plain-language README explaining the ideas before any code
- Visualizations wherever possible
- Comments that explain *why*, not just *what*
- A "what to try next" section for extending the work

## Getting started
Each experiment has its own `requirements.txt` and `README.md`. Clone the repo, navigate to whichever folder interests you, and follow the instructions there.

```bash
git clone https://github.com/lijo8146/systems-at-play.git
cd systems-at-play/catan-probability-models
pip install -r requirements.txt
python run_analysis.py
```

A shared utility library lives in [`shared-utils/`](./shared-utils/) with common statistical helpers used across experiments.

## Contributing
Contributions are welcome, especially new experiments, better explanations, and student questions that become documented answers. See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to add your own experiment to the collection.

## License
MIT : see [LICENSE](./LICENSE). Use freely, teach freely, build on freely.

## Citation
If you use this in a course, paper, or project, a citation is appreciated but not required. See [CITATION.cff](./CITATION.cff).

*Built with Python · NumPy · Matplotlib · NetworkX · Streamlit*

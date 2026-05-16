# Pandemic Spread Simulator
A stochastic network-based pandemic simulator inspired by the board game *Pandemic*. Models disease spread across a weighted city graph using a discrete-time SIR framework, with player roles that intervene in the system then analyzes their effectiveness via Monte Carlo simulation.

## What it models
Cities are nodes in a weighted graph. Each edge represents a travel route where higher weight means more passenger movement and faster cross-city spread. The disease evolves each "week" (step) through:

1. **Within-city transmission** : stochastic binomial draws based on the local infection rate
2. **Travel-based seeding** : infectious travellers import cases proportional to edge weight
3. **Recovery** : binomial draws at a fixed recovery rate
4. **Cure research** : progresses automatically (roles can accelerate it)

### Roles
Each role modifies the simulation's parameters:

| Role | Effect |
|---|---|
| **Medic** | −60 % local spread in stationed city; can mass-treat 200 citizens |
| **Scientist** | 2.5× cure research speed |
| **Epidemiologist** | −25 % spread rate globally |
| **Dispatcher** | −15 % spread rate globally via travel coordination |
| **Quarantine Specialist** | Can lock down a city, blocking all travel spread |

## Getting started

```bash
git clone https://github.com/yourusername/pandemic-simulator.git
cd pandemic-simulator
pip install -r requirements.txt
```

### Interactive Streamlit app

```bash
streamlit run app.py
```

The dashboard gives you:
- **Role picker** : toggle any combination of the 5 player roles in the sidebar
- **Animated network** : watch the city graph light up red step-by-step, with Plotly hover for per-city stats
- **Play/Pause/Scrub** : step through frames at any speed with the slider
- **Live timeline** : per-city infection curves and cure-progress update as the animation plays
- **Monte Carlo panel** : run a quick role comparison (50–500 sims) from inside the app

### Run a single simulation (CLI)

```bash
# No roles (baseline)
python run_simulation.py

# With a Scientist and Medic stationed in Beijing
python run_simulation.py --roles scientist medic

# Try the Quarantine Specialist with a specific seed
python run_simulation.py --roles quarantine_specialist scientist --seed 7
```

### Monte Carlo analysis (CLI)
Compare all role configurations across 300 independent simulations:

```bash
python run_simulation.py --monte-carlo --n 300
```

### Jupyter notebook

Generate and open the analysis notebook:

```bash
python generate_notebook.py     # creates simulation_notebook.ipynb
jupyter notebook simulation_notebook.ipynb
```

The notebook covers single-run walkthroughs, network snapshots, parameter sensitivity sweeps, Monte Carlo role comparison, and a section on key modelling insights.

## Output examples
**Network state** (`output/network_final.png`)  
Nodes coloured yellow to red by infection rate. Node size ∝ population. Edge thickness approximate travel weight. Blue outlines indicate quarantined cities.

**Spread timeline** (`output/timeline.png`)  
Per-city infected curves + cure-progress fill chart over simulation steps.

**Role comparison** (`output/role_comparison_winrate.png`)  
Horizontal bar chart ranking role configurations by win rate across N simulations.

## Project structure

```
pandemic-simulator/
├── app.py                      # Streamlit interactive dashboard
├── run_simulation.py           # CLI entry point
├── generate_notebook.py        # Builds simulation_notebook.ipynb
├── simulation_notebook.ipynb   # Jupyter analysis notebook
├── simulation/
│   ├── network.py              # City dataclass and world graph
│   ├── infection.py            # SIR simulator engine
│   └── roles.py                # Player role definitions
├── analysis/
│   └── monte_carlo.py          # Repeated-trial role effectiveness analysis
├── visualization/
│   └── plots.py                # Network snapshots, timelines, comparison charts
└── requirements.txt
```

## Important modelling decisions
- **Stochastic not deterministic** : every run uses `numpy.random.default_rng` with a settable seed, so results are reproducible but still reflect real uncertainty
- **Discrete-time SIR** : chosen over continuous ODE models because it maps naturally to board-game mechanics (turns/steps) while still being mathematically grounded
- **Binomial spread** : `Binomial(S, β·I/N)` captures demographic stochasticity better than a simple rate equation at the city scale
- **Travel seeding** : a small fraction of each city's susceptible population is exposed to infectious travellers each step, scaled by edge weight and source infection rate

## Extending the project
Some directions if you want to go further:

- **SEIR model** : add an Exposed compartment for a latent period
- **Mutation events** : random mid-game increases to `base_spread_rate`
- **Streamlit dashboard** : real-time interactive controls for role assignment
- **Animated GIF output** : frame-by-frame network spread visualization
- **Real city data** : swap the toy network for IATA flight data

## Inspiration
Board game mechanics are underrated as a framework for teaching systems dynamics. *Pandemic* is essentially a cooperative optimization problem over a stochastic graph which is the same structure that appears in epidemiology, supply chain risk, and environmental spread modeling.

*Built with Python · NetworkX · NumPy · Matplotlib*

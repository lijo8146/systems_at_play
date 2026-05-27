# Contributing to Systems at Play

Contributions are genuinely welcome.

## The easiest contributions
- **Fix a confusing explanation.** If something in a README or code comment didn't make sense, a clearer version is valuable.
- **Add a visualization.** Charts and diagrams make everything more accessible.
- **Add a "what to try" idea.** Did you extend an experiment in an interesting direction? Document it.
- **Report bugs.** Open an issue if something doesn't run or gives unexpected results.

## Adding a new experiment
New experiments are the most exciting contributions. A good experiment:

1. **Starts with a question** : "what actually happens if…?" or "how does X affect Y?"
2. **Uses a game, puzzle, or simulation** as the entry point
3. **Connects to a real idea** : probability, graph theory, optimization, AI, etc.
4. **Is readable** : someone new to the topic should be able to follow along

### Structure for a new experiment
Create a folder at the top level following this pattern:

```
your-experiment-name/
├── README.md          : explain the question and the ideas (required)
├── requirements.txt   : keep dependencies minimal
├── your_code.py       : well-commented, readable code
└── run_analysis.py    : a simple entry point that produces output
```

The `README.md` should:
- Open with the question the experiment explores (not "this module implements…")
- Explain the key ideas in plain language before any code
- Include at least one example output or visualization
- Close with a "what to try" section suggesting extensions

### Adding it to the collection
1. Fork the repo and create a branch
2. Add your experiment folder
3. Add a row to the table in the top-level `README.md`
4. Open a pull request with a short description of what the experiment explores

## Code style
- **Clarity over cleverness.** This is a teaching repo so readable code matters more than optimized code.
- **Comments explain why, not what.** Assume readers know Python syntax; help them understand the thinking.
- **Type hints are encouraged** but not required.
- **Visualizations are encouraged** matplotlib, plotly, or networkx all work fine.

## Questions?
Open an issue and ask. There are no bad questions, especially from students working through these examples for the first time.

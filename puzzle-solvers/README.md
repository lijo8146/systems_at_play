# Puzzle Solvers

> *Every puzzle is a search problem. The question is how smart your search is.*

## The question
How does a computer solve a Sudoku? How does it find its way through a maze?

Both problems look very different on the surface, but they share the same underlying structure: you have a space of possible states, a set of rules about which moves are legal, and you're trying to find a path from "unsolved" to "solved."

This module implements two classic puzzle solvers with explicit, readable reasoning so you can see *how* the thinking works, not just *that* it works.

## Sudoku backtracking with constraint propagation
**Pure backtracking** is simple: try a digit, if it causes a conflict, try the next one. This works but is slow so it's a hard puzzle might require thousands of failed guesses.

**Constraint propagation** adds a smarter pre-step: before guessing, eliminate any digit that's already *provably* impossible based on the current board state. This dramatically reduces the search space.

The combination is to propagate constraints first, then backtrack when stuck. This solves most published Sudoku puzzles in milliseconds.

### Run it

```bash
python sudoku.py                    
python sudoku.py --puzzle "your81charstring"
```

## Maze generation and solving
A maze is a graph. Rooms are nodes; passages are edges. We use two algorithms:
**Generation:** Recursive backtracking (also called depth-first search with random wall removal). This creates *perfect mazes* (one that has exactly one path between any two cells, no loops).

**Solving:**
- **BFS** (breadth-first search) finds the *shortest* path
- **DFS** (depth-first search) finds *a* path but not necessarily the shortest

Comparing BFS and DFS on the same maze is a great way to see why algorithm choice matters.

### Run it
```bash
python maze.py                      
python maze.py --width 25 --height 25 --seed 7
```

## What to try
- **Sudoku:** Add a "difficulty estimator" that scores puzzles by how often backtracking is needed
- **Sudoku:** Implement the "naked pairs" or "hidden singles" techniques used by human solvers
- **Maze:** Try A* search (heuristic-guided) and compare path lengths to BFS
- **Maze:** Generate a maze with multiple entry/exit points and find the optimal route
- **Both:** Visualise the search process step-by-step to watch the algorithm think

"""
maze.py maze generation (recursive backtracking) and solving (BFS + DFS).

Every maze is a graph problem
A grid maze is just a graph where:
- Nodes = cells
- Edges = open passages between adjacent cells

Generation (recursive backtracking):
  Start at a random cell. Visit unvisited neighbours in random order,
  carving a passage to each. This produces a *perfect maze*: exactly
  one path between any two cells with no loops, no isolated regions.

Solving:
  BFS (breadth-first search):
    Explores cells in order of distance from the start.
    Guarantees the *shortest* path.

  DFS (depth-first search):
    Plunges deep before backtracking.
    Finds a path but not necessarily the shortest.

Comparing their paths on the same maze shows why algorithm choice matters.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Types

class Cell(NamedTuple):
    row: int
    col: int

# cell to set of reachable neighbours
Maze = dict[Cell, set[Cell]]   


# Maze generation

def generate_maze(
    width: int,
    height: int,
    seed: int | None = None,
    extra_passages: int = 0,
) -> Maze:
    """
    Generate a maze using recursive backtracking (iterative version).

    Parameters
    extra_passages : if > 0, randomly remove this many walls after generation
                     to create loops. A "perfect" maze (no loops) has
                     exactly one path between any two cells — so BFS and DFS
                     always find the same path. Add loops to see them diverge.

    Returns an adjacency dict: cell to set of open neighbours.
    """
    rng = random.Random(seed)
    cells = {Cell(r, c) for r in range(height) for c in range(width)}

    maze: Maze = {cell: set() for cell in cells}
    visited: set[Cell] = set()

    def grid_neighbours(cell: Cell) -> list[Cell]:
        r, c = cell
        candidates = [Cell(r-1,c), Cell(r+1,c), Cell(r,c-1), Cell(r,c+1)]
        return [n for n in candidates if n in cells]

    # Iterative DFS for generation
    start = Cell(0, 0)
    stack = [start]
    visited.add(start)

    while stack:
        current = stack[-1]
        unvisited = [n for n in grid_neighbours(current) if n not in visited]
        if unvisited:
            chosen = rng.choice(unvisited)
            maze[current].add(chosen)
            maze[chosen].add(current)
            visited.add(chosen)
            stack.append(chosen)
        else:
            stack.pop()

    # Add random extra passages to create loops
    if extra_passages > 0:
        removed = 0
        all_cells = list(cells)
        rng.shuffle(all_cells)
        for cell in all_cells:
            if removed >= extra_passages:
                break
            for neighbour in grid_neighbours(cell):
                if neighbour not in maze[cell]:
                    maze[cell].add(neighbour)
                    maze[neighbour].add(cell)
                    removed += 1
                    break

    return maze


# Solving algorithms

def bfs(maze: Maze, start: Cell, goal: Cell) -> tuple[list[Cell], list[Cell]]:
    """
    Breadth-first search so it finds the *shortest* path from start to goal.

    BFS explores cells in rings of increasing distance from the start,
    so the first time it reaches the goal, it's via the shortest route.

    Returns (path, exploration_order) where exploration_order shows which cells
    were visited and in what order, revealing the "wave" expansion pattern.
    """
    queue: deque[list[Cell]] = deque([[start]])
    visited: set[Cell] = {start}
    explored: list[Cell] = [start]

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == goal:
            return path, explored

        for neighbour in sorted(maze[current]):  
            if neighbour not in visited:
                visited.add(neighbour)
                explored.append(neighbour)
                queue.append(path + [neighbour])

    return [], explored


def dfs(maze: Maze, start: Cell, goal: Cell) -> tuple[list[Cell], list[Cell]]:
    """
    Depth-first search finds a path, but not necessarily the shortest.

    DFS dives deep along one branch before backtracking, which often
    produces a longer, more winding path than BFS on mazes with loops.

    Returns (path, exploration_order).
    """
    stack: list[list[Cell]] = [[start]]
    visited: set[Cell] = {start}
    explored: list[Cell] = [start]

    while stack:
        path = stack.pop()
        current = path[-1]

        if current == goal:
            return path, explored

        for neighbour in sorted(maze[current]):
            if neighbour not in visited:
                visited.add(neighbour)
                explored.append(neighbour)
                stack.append(path + [neighbour])

    return [], explored


# Visualisation

def plot_maze(
    maze: Maze,
    width: int,
    height: int,
    bfs_result: tuple[list[Cell], list[Cell]] | None = None,
    dfs_result: tuple[list[Cell], list[Cell]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Render the maze with BFS and DFS paths and exploration heatmaps.

    Left panel:  BFS exploration order shown as colour gradient (wave pattern)
    Right panel: DFS exploration order shown as colour gradient (deep dive pattern)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    configs = [
        ("BFS : shortest path\n(explores in expanding waves)", bfs_result, plt.cm.Blues,  "#1565C0"),
        ("DFS : first path found\n(dives deep before backtracking)", dfs_result, plt.cm.Oranges, "#E65100"),
    ]

    for ax, (title, result, cmap, path_color) in zip(axes, configs):
        path, explored = result if result else ([], [])

        ax.set_xlim(-0.05, width + 0.05)
        ax.set_ylim(-0.05, height + 0.05)
        ax.set_aspect("equal")
        ax.axis("off")

        n_explored = len(explored)

        # Draw exploration heatmap (order visited maps to colour intensity)
        explore_order = {cell: i for i, cell in enumerate(explored)}
        for r in range(height):
            for c in range(width):
                cell = Cell(r, c)
                if cell in explore_order:
                    intensity = explore_order[cell] / max(n_explored, 1)
                    colour = cmap(0.15 + 0.75 * intensity)
                    x, y = c, height - r - 1
                    rect = mpatches.Rectangle((x, y), 1, 1, color=colour, alpha=0.55, zorder=0)
                    ax.add_patch(rect)

        # Draw walls
        for r in range(height):
            for c in range(width):
                cell = Cell(r, c)
                x, y = c, height - r - 1
                if Cell(r-1, c) not in maze.get(cell, set()):
                    ax.plot([x, x+1], [y+1, y+1], "k-", linewidth=1.2, zorder=2)
                if Cell(r+1, c) not in maze.get(cell, set()):
                    ax.plot([x, x+1], [y, y],     "k-", linewidth=1.2, zorder=2)
                if Cell(r, c-1) not in maze.get(cell, set()):
                    ax.plot([x, x], [y, y+1],     "k-", linewidth=1.2, zorder=2)
                if Cell(r, c+1) not in maze.get(cell, set()):
                    ax.plot([x+1, x+1], [y, y+1], "k-", linewidth=1.2, zorder=2)

        # Outer border
        for x0,y0,x1,y1 in [(0,0,width,0),(0,height,width,height),(0,0,0,height),(width,0,width,height)]:
            ax.plot([x0,x1],[y0,y1],"k-", linewidth=2, zorder=3)

        # Draw solution path
        if path:
            xs = [c + 0.5 for _, c in path]
            ys = [height - r - 0.5 for r, _ in path]
            ax.plot(xs, ys, color=path_color, linewidth=2.5, alpha=0.9, zorder=4)
            ax.scatter([xs[0]], [ys[0]], c="limegreen", s=150, zorder=5)
            ax.scatter([xs[-1]], [ys[-1]], c="red",      s=150, zorder=5)

        path_len = len(path)
        ax.set_title(
            f"{title}\n{path_len} steps · explored {n_explored}/{width*height} cells",
            fontsize=10, fontweight="bold",
        )

    bfs_path = bfs_result[0] if bfs_result else []
    dfs_path = dfs_result[0] if dfs_result else []
    extra = len(dfs_path) - len(bfs_path)
    fig.suptitle(
        f"Maze {width}×{height}  ·  BFS: {len(bfs_path)} steps  ·  "
        f"DFS: {len(dfs_path)} steps  ·  "
        f"Path overhead: {'+' if extra >= 0 else ''}{extra} steps",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maze generator and solver")
    parser.add_argument("--width",           type=int, default=20)
    parser.add_argument("--height",          type=int, default=20)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--extra-passages",  type=int, default=30,
                        help="Extra wall removals to create loops (0 = perfect maze)")
    parser.add_argument("--save",            type=str, default="output/maze.png")
    args = parser.parse_args()

    import os; os.makedirs("output", exist_ok=True)
    import matplotlib; matplotlib.use("Agg")

    perfect = args.extra_passages == 0
    maze_type = "perfect" if perfect else f"imperfect ({args.extra_passages} extra passages)"
    print(f"Generating {args.width}×{args.height} {maze_type} maze (seed={args.seed})…")
    if perfect:
        print("  Note: perfect mazes have exactly one path between any two cells,")
        print("  so BFS and DFS always find the same path length. The interesting")
        print("  difference is their *exploration pattern* shown as the colour heatmap.")

    maze = generate_maze(args.width, args.height,
                         seed=args.seed, extra_passages=args.extra_passages)

    start = Cell(0, 0)
    goal  = Cell(args.height - 1, args.width - 1)

    bfs_path, bfs_explored = bfs(maze, start, goal)
    dfs_path, dfs_explored = dfs(maze, start, goal)

    print(f"BFS: {len(bfs_path)} steps, explored {len(bfs_explored)} cells")
    print(f"DFS: {len(dfs_path)} steps, explored {len(dfs_explored)} cells")
    print(f"Path overhead: {len(dfs_path) - len(bfs_path):+d} steps")

    fig = plot_maze(maze, args.width, args.height,
                    bfs_result=(bfs_path, bfs_explored),
                    dfs_result=(dfs_path, dfs_explored),
                    save_path=args.save)
    print(f"Saved → {args.save}")

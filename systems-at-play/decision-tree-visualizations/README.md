# Decision Tree Visualizations

> *Every strategic decision is a node in a tree. The question is how deep you're willing to look.*

## The question

When a computer plays a game, it doesn't "think" the way people think, it searches. It constructs a tree of possible moves and countermoves, evaluates the outcomes at the leaves, and picks the branch that leads to the best outcome *assuming the opponent plays perfectly too*.

This is the **minimax algorithm**, and it underlies everything from chess engines to game AI research.

This module visualises that search process: what does the tree actually look like? How big does it get? Where does pruning help?

## Experiments (coming soon)

### 1. Tic-tac-toe minimax (`tictactoe.py`)
The complete game tree for tic-tac-toe has ~255,168 leaf nodes. Minimax can solve it perfectly. A good entry point because:
- The game is small enough to visualise the full tree
- Optimal play always leads to a draw (provably)

### 2. Alpha-beta pruning (`pruning.py`)
Alpha-beta pruning is the key insight that makes minimax practical. With pruning, you can safely ignore branches that can't affect the final decision. On a well-ordered tree, it reduces the nodes searched from O(b^d) to O(b^(d/2)).

### 3. Connect Four heuristics (`connect_four.py`)
Connect Four's tree is too large to search completely (game tree complexity ~10^21). Instead, we search to a fixed depth and use a *heuristic evaluation function* to estimate position quality. Visualising this shows the tradeoffs between search depth and computation time.


## What to try
- Visualise the full tic-tac-toe game tree as a network diagram with NetworkX
- Measure how pruning efficiency changes with move ordering (best moves first vs random)
- Compare a random player vs a 2-ply minimax player vs a 4-ply minimax player on Connect Four
- Plot "nodes searched" as a function of search depth to see the exponential growth

"""
puzzles.py classic logic grid puzzles for the solver demo.

Clue helper functions handle partial assignments correctly:
  - Return True if the relevant items are not yet assigned (not violated)
  - Return False only when the clue is DEFINITIVELY violated
  - Never return False just because a second attribute is unassigned
"""

from __future__ import annotations
from solver import LogicPuzzle


# Clue helper functions

def same_entity(assignment, puzzle, cat_a, item_a, cat_b, item_b):
    """The entity with item_a in cat_a also has item_b in cat_b."""
    ia = puzzle.item_index(cat_a, item_a)
    ib = puzzle.item_index(cat_b, item_b)
    ca = puzzle.cat_index(cat_a)
    cb = puzzle.cat_index(cat_b)
    for e in range(puzzle.n):
        if assignment.get((e, ca)) == ia:
            val_b = assignment.get((e, cb))
            if val_b is None:
                return True   # cat_b not yet assigned 
            return val_b == ib
    return True   # item_a not yet assigned


def not_same_entity(assignment, puzzle, cat_a, item_a, cat_b, item_b):
    """The entity with item_a does NOT have item_b."""
    ia = puzzle.item_index(cat_a, item_a)
    ib = puzzle.item_index(cat_b, item_b)
    ca = puzzle.cat_index(cat_a)
    cb = puzzle.cat_index(cat_b)
    for e in range(puzzle.n):
        if assignment.get((e, ca)) == ia:
            val_b = assignment.get((e, cb))
            if val_b is None:
                return True   # not yet assigned 
            return val_b != ib
    return True


def next_to(assignment, puzzle, cat_a, item_a, cat_b, item_b):
    """Entities with item_a and item_b are adjacent (differ by 1)."""
    ia = puzzle.item_index(cat_a, item_a)
    ib = puzzle.item_index(cat_b, item_b)
    ca = puzzle.cat_index(cat_a)
    cb = puzzle.cat_index(cat_b)
    ea = eb = None
    for e in range(puzzle.n):
        if assignment.get((e, ca)) == ia:
            ea = e
        if assignment.get((e, cb)) == ib:
            eb = e
    if ea is None or eb is None:
        return True   # one or both not yet placed
    return abs(ea - eb) == 1


def immediately_left(assignment, puzzle, cat_a, item_a, cat_b, item_b):
    """Entity with item_a has index exactly one less than entity with item_b."""
    ia = puzzle.item_index(cat_a, item_a)
    ib = puzzle.item_index(cat_b, item_b)
    ca = puzzle.cat_index(cat_a)
    cb = puzzle.cat_index(cat_b)
    ea = eb = None
    for e in range(puzzle.n):
        if assignment.get((e, ca)) == ia:
            ea = e
        if assignment.get((e, cb)) == ib:
            eb = e
    if ea is None or eb is None:
        return True
    return ea == eb - 1


def at_position(assignment, puzzle, cat, item, pos):
    """Entity at position pos has item in cat."""
    item_idx = puzzle.item_index(cat, item)
    cat_idx  = puzzle.cat_index(cat)
    val = assignment.get((pos, cat_idx))
    return val is None or val == item_idx


# Puzzle 1: 3x3 Colors

def make_colors_puzzle():
    puzzle = LogicPuzzle(
        n=3, entity_name="house",
        categories=["color", "owner", "pet"],
        items={
            "color": ["red", "blue", "green"],
            "owner": ["Alice", "Bob", "Carol"],
            "pet":   ["cat", "dog", "fish"],
        },
    )
    puzzle.clues = [
        ("Alice lives in the red house.",
         lambda a, p: same_entity(a, p, "owner","Alice","color","red")),
        ("The blue house is immediately left of the green house.",
         lambda a, p: immediately_left(a, p, "color","blue","color","green")),
        ("Bob has a dog.",
         lambda a, p: same_entity(a, p, "owner","Bob","pet","dog")),
        ("The cat owner lives in the blue house.",
         lambda a, p: same_entity(a, p, "pet","cat","color","blue")),
        ("Carol does not live in the green house.",
         lambda a, p: not_same_entity(a, p, "owner","Carol","color","green")),
    ]
    return puzzle


# Puzzle 2: 4x4 Weekend Plans

def make_weekend_puzzle():
    puzzle = LogicPuzzle(
        n=4, entity_name="person",
        categories=["name", "day", "activity", "snack"],
        items={
            "name":     ["Alex", "Blake", "Casey", "Dana"],
            "day":      ["Friday", "Saturday", "Sunday", "Monday"],
            "activity": ["hiking", "reading", "cooking", "gaming"],
            "snack":    ["chips", "fruit", "nuts", "popcorn"],
        },
    )
    puzzle.clues = [
        ("Alex goes hiking.",
         lambda a, p: same_entity(a, p, "name","Alex","activity","hiking")),
        ("The person free on Saturday goes gaming.",
         lambda a, p: same_entity(a, p, "day","Saturday","activity","gaming")),
        ("Blake eats popcorn.",
         lambda a, p: same_entity(a, p, "name","Blake","snack","popcorn")),
        ("The hiker eats nuts.",
         lambda a, p: same_entity(a, p, "activity","hiking","snack","nuts")),
        ("Casey is free on Saturday.",
         lambda a, p: same_entity(a, p, "name","Casey","day","Saturday")),
        ("The cook is free on Sunday.",
         lambda a, p: same_entity(a, p, "activity","cooking","day","Sunday")),
        ("Dana does not eat chips.",
         lambda a, p: not_same_entity(a, p, "name","Dana","snack","chips")),
        ("The reader is free on Monday.",
         lambda a, p: same_entity(a, p, "activity","reading","day","Monday")),
    ]
    return puzzle


# Puzzle 3: 5x5 Zebra/Einstein 

def make_zebra_puzzle():
    puzzle = LogicPuzzle(
        n=5, entity_name="house",
        categories=["color","nationality","drink","smoke","pet"],
        items={
            "color":       ["red","green","white","yellow","blue"],
            "nationality": ["English","Swedish","Danish","Norwegian","German"],
            "drink":       ["tea","coffee","milk","beer","water"],
            "smoke":       ["Pall Mall","Dunhill","Blend","BlueMaster","Prince"],
            "pet":         ["dog","bird","cat","horse","fish"],
        },
    )
    puzzle.clues = [
        ("The Englishman lives in the red house.",
         lambda a, p: same_entity(a, p, "nationality","English","color","red")),
        ("The Swede keeps a dog.",
         lambda a, p: same_entity(a, p, "nationality","Swedish","pet","dog")),
        ("The Dane drinks tea.",
         lambda a, p: same_entity(a, p, "nationality","Danish","drink","tea")),
        ("The green house is immediately left of the white house.",
         lambda a, p: immediately_left(a, p, "color","green","color","white")),
        ("The green house owner drinks coffee.",
         lambda a, p: same_entity(a, p, "color","green","drink","coffee")),
        ("The Pall Mall smoker keeps a bird.",
         lambda a, p: same_entity(a, p, "smoke","Pall Mall","pet","bird")),
        ("The yellow house owner smokes Dunhill.",
         lambda a, p: same_entity(a, p, "color","yellow","smoke","Dunhill")),
        ("The center house owner drinks milk.",
         lambda a, p: at_position(a, p, "drink","milk", 2)),
        ("The Norwegian lives in the first house.",
         lambda a, p: at_position(a, p, "nationality","Norwegian", 0)),
        ("The Blend smoker lives next to the cat owner.",
         lambda a, p: next_to(a, p, "smoke","Blend","pet","cat")),
        ("The horse owner lives next to the Dunhill smoker.",
         lambda a, p: next_to(a, p, "pet","horse","smoke","Dunhill")),
        ("The BlueMaster smoker drinks beer.",
         lambda a, p: same_entity(a, p, "smoke","BlueMaster","drink","beer")),
        ("The German smokes Prince.",
         lambda a, p: same_entity(a, p, "nationality","German","smoke","Prince")),
        ("The Norwegian lives next to the blue house.",
         lambda a, p: next_to(a, p, "nationality","Norwegian","color","blue")),
        ("The Blend smoker has a neighbor who drinks water.",
         lambda a, p: next_to(a, p, "smoke","Blend","drink","water")),
    ]
    return puzzle


# Registry

PUZZLES = {
    "3x3 Colors (easy)":     make_colors_puzzle,
    "4x4 Weekend Plans":     make_weekend_puzzle,
    "5x5 Zebra / Einstein":  make_zebra_puzzle,
}

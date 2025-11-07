import random
from grid_universe import renderer
from grid_universe.gym_env import (
    Observation,
    Action,
    BaseAction,
    step
)
from grid_universe.levels.grid import Level

from typing import List, Tuple

# Core API
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.levels.convert import to_state, from_state

# Factories
from grid_universe.levels.factories import (
    create_floor, create_agent, create_box, create_coin, create_exit, create_wall,
    create_key, create_door, create_portal, create_core, create_hazard, create_monster,
    create_phasing_effect, create_speed_effect, create_immunity_effect,
)

from grid_universe.components.properties.appearance import AppearanceName
from grid_universe.components.properties import MovingAxis

from grid_universe.utils.inventory import (
    has_key_with_id 
)

# Movement and objectives
from grid_universe.moves import default_move_fn
from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn
)

def build_level_speed_test():
    level = Level(
        width=14,
        height=3,
        move_fn=default_move_fn,           # choose movement semantics
        objective_fn=default_objective_fn,    # win when stand on exit
        seed=9,                         # for reproducibility
    )

    # 2) Layout: floors, then place objects
    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount = 3))

    level.add((1, 1), create_agent(health=5))

    level.add((0, 1), create_speed_effect(multiplier=2, time=5))

    level.add((13, 1), create_exit())

    return to_state(level)

def build_random_level_with_coins(seed: int = None) -> State:
    """
    Builds a random level layout with coins, walls, agent, and exit.
    The layout is procedurally generated but reproducible via seed.
    """
    w, h = 8, 6
    random.seed(seed)

    level = Level(
        w,
        h,
        move_fn=default_move_fn,
        objective_fn=exit_objective_fn,
        seed=seed,
        turn_limit=None,
    )

    for y in range(level.height):
        for x in range(level.width):
            level.add((x, y), create_floor(cost_amount = 3))

    # Add agent at top-left corner
    level.add((1, 1), create_agent(health=5))

    # Add exit at bottom-right corner
    level.add((w - 2, h - 2), create_exit())

    # Scatter coins randomly across empty tiles
    num_coins = random.randint(3, 8)
    for _ in range(num_coins):
        x, y = random.randint(1, w - 2), random.randint(1, h - 2)
        
        level.add((x, y), create_coin(reward=5))

    def build_level():
        return to_state(level)

    return build_level

def build_level_hard_AF() -> Level:
    level = Level(width=9, height=7, move_fn=default_move_fn, objective_fn=default_objective_fn, seed=0)

    level.add((0, 0), create_agent(health=5))
    for x, y in [(6, 2), (2, 5)]:
        level.add((x, y), create_coin(reward=5))
    for x, y in [(8, 3), (0, 4)]:
        level.add((x, y), create_core(reward=0, required=True))
    level.add((0, 5), create_door('A'))
    level.add((3, 6), create_exit())
    for x, y in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6)]:
        level.add((x, y), create_floor(cost_amount=3))
    level.add((7, 3), create_hazard(AppearanceName.LAVA, 2, False))
    for x, y in [(0, 3), (8, 5)]:
        level.add((x, y), create_hazard(AppearanceName.SPIKE, 2, False))
    level.add((3, 2), create_key('A'))
    level.add((7, 1), create_monster(damage=1, lethal=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=-1, moving_bounce=True, moving_speed=1))
    level.add((3, 3), create_monster(damage=1, lethal=False, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1, moving_bounce=True, moving_speed=1))
    level.add((8, 6), create_phasing_effect(time=5))
    level.add((4, 3), create_speed_effect(multiplier=2, time=5))
    for x, y in [(1, 0), (3, 1), (5, 1), (1, 2), (2, 2), (5, 2), (1, 3), (1, 4), (5, 4), (6, 4), (7, 4), (8, 4), (3, 5), (4, 5), (2, 6), (5, 6)]:
        level.add((x, y), create_wall())
    p1 = create_portal()
    p2 = create_portal(pair=p1)
    level.add((7, 2), p1)
    level.add((7, 5), p2)

    return to_state(level)
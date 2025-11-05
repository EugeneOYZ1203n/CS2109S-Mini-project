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
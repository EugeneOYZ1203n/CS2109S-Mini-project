# Core API
import os
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.levels.convert import to_state, from_state

# Factories
from grid_universe.levels.factories import (
    create_floor, create_agent, create_box, create_coin, create_exit, create_wall,
    create_key, create_door, create_portal, create_core, create_hazard, create_monster,
    create_phasing_effect, create_speed_effect, create_immunity_effect,
)

# Movement and objectives
from grid_universe.moves import default_move_fn
from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn
)

from grid_universe.renderer.texture import TextureRenderer
from IPython.display import display

from typing import Callable
from dataclasses import replace

from grid_universe.components.properties.appearance import AppearanceName
from grid_universe.components.properties import MovingAxis

from grid_universe.gym_env import GridUniverseEnv

ASSET_ROOT = "data/assets/"

from PIL import Image

def create_env(builder: Callable[[], State], seed: int = 42,  turn_limit: int | None = None, **kwargs) -> GridUniverseEnv:
    sample_state = builder()
    def _initial_state_fn(*args, **kwargs) -> State:
        state = builder()
        if turn_limit is not None:
            state = replace(state, turn_limit=turn_limit)
        return replace(state, seed=seed)
    return GridUniverseEnv(initial_state_fn=_initial_state_fn, width=sample_state.width, height=sample_state.height, render_asset_root=ASSET_ROOT, **kwargs)

def build_level_with_single_entity(entity, seed = 42, is_floor = True):
    level = Level(
        width=2,
        height=1,
        move_fn=default_move_fn,           # choose movement semantics
        objective_fn=default_objective_fn,    # win when stand on exit
        seed=seed,                         # for reproducibility
    )

    if is_floor:
        level.add((0, 0), create_floor(cost_amount = 3))
    level.add((0, 0), entity)
    level.add((1, 0), create_agent(health=5))

    def build_level():
        return to_state(level)

    return build_level

entities = {
    "agent" : create_agent(health=5), #Can vary per level, need to check
    "coin" : create_coin(reward=5), 
    "core" : create_core(reward=None, required=True), 
    "monster_horizontal_neg" : create_monster(damage=1, moving_axis=MovingAxis.HORIZONTAL, moving_direction=-1), 
    "monster_vertical_neg" : create_monster(damage=1, moving_axis=MovingAxis.VERTICAL, moving_direction=-1), 
    "monster_horizontal_pos" : create_monster(damage=1, moving_axis=MovingAxis.HORIZONTAL, moving_direction=1), 
    "monster_vertical_pos" : create_monster(damage=1, moving_axis=MovingAxis.VERTICAL, moving_direction=1), 
    "lava" : create_hazard(AppearanceName.LAVA, 2, False), 
    "spike" : create_hazard(AppearanceName.SPIKE, 2, False),
    "box_static" : create_box(pushable=True),
    "box_horizontal_neg" : create_box(moving_axis=MovingAxis.HORIZONTAL, moving_direction=-1),
    "box_vertical_neg" : create_box(moving_axis=MovingAxis.VERTICAL, moving_direction=-1),
    "box_horizontal_pos" : create_box(moving_axis=MovingAxis.HORIZONTAL, moving_direction=-1),
    "box_vertical_pos" : create_box(moving_axis=MovingAxis.VERTICAL, moving_direction=-1),
    "key" : create_key(key_id="XXX"),
    "door" : create_door(key_id="XXX"),
    "portal" : create_portal(), #Pair entity property needs to be set after the first pair is found
    "immunity_effect" : create_immunity_effect(time=5),
    "phasing_effect" : create_phasing_effect(time=5),
    "speed_effect" : create_speed_effect(multiplier=2, time=5),
    "exit" : create_exit()
}

renderer1 = TextureRenderer(resolution=48)
renderer2 = TextureRenderer(resolution=96)
renderer2 = TextureRenderer(resolution=54)



for key in entities.keys():
    seed = 2
    is_floor = key in {"lava", "spike"}
    level = build_level_with_single_entity(entities[key], seed=seed, is_floor=False)

    image = renderer1.render(level())

    # Get width and height
    width, height = image.size

    # Crop to the left half: (left, upper, right, lower)
    left_half = image.crop((0, 0, width // 2, height))

    save_dir = os.path.join("new_images", key)
    os.makedirs(save_dir, exist_ok=True)  # create folder if it doesn't exist

    filename = f"{key}_{seed}.png"
    save_path = os.path.join(save_dir, filename)

    left_half.save(save_path)
    print(f"Saved: {save_path}")
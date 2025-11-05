from grid_universe.gym_env import GridUniverseEnv, Observation, Action
from grid_universe.state import State

from cipher_text_classifier import Agent

from grid_universe.renderer.texture import TextureRenderer
from utils import show_images_interactive

renderer = TextureRenderer(resolution=480)


def evaluate(agent_class: type[Agent], env: GridUniverseEnv) -> tuple[float, bool, bool, list[tuple[State,Action]]]:
    """Utility to execute a full episode using the provided agent class.

    Parameters
    ----------
    agent_class : type[Agent]
        Class (not instance). The function will instantiate it via agent_class().
    env : GridUniverseEnv
        An initialized GridUniverseEnv environment. The caller is responsible
        for any seeding / configuration before passing it here.

    Returns
    -------
    total_reward : float
        Cumulative reward collected until termination.
    win : bool
        True if the agent reached a win condition.
    lose : bool
        True if the agent triggered a loss condition.

    Notes
    -----
    - This helper mirrors the Coursemology evaluation loop structure.
    """
    state, _ = env.reset()
    agent = agent_class()
    win = False
    lose = False
    total_reward = 0
    history: list[tuple[State,Action]] = []
    while not (win or lose):
        action = agent.step(state)

        # record state+action pairs for analysis / debugging
        assert env.state is not None
        history.append((env.state, action))

        state, reward, win, lose, _ = env.step(action)
        total_reward += reward

    assert env.state is not None
    history.append((env.state, Action.WAIT))  # final state
    
    return total_reward, win, lose, history

import pandas as pd
import time
from typing import Callable, Iterable
from dataclasses import replace
from pprint import pprint
from grid_universe.examples.cipher_objective_levels import CipherObjectivePair, to_cipher_level, patch_env_redact_objective_fn
from grid_universe.examples.gameplay_levels import (
    build_level_maze_turns, build_level_optional_coin, build_level_required_one, build_level_required_two, 
    build_level_key_door, build_level_hazard_detour, build_level_portal_shortcut, build_level_pushable_box, build_level_enemy_patrol, 
    build_level_power_shield, build_level_power_ghost, build_level_power_boots, build_level_capstone
)
from utils import get_level_name, get_minimum_total_reward


# Change this if you want to use custom game assets or a different cipher text file.
ASSET_ROOT = "data/assets/"
CIPHERTEXT_PATH = "data/cipher_objective.csv"

GAMEPLAY_LEVEL_BUILDERS = [
    build_level_maze_turns, build_level_optional_coin, build_level_required_one, build_level_required_two, 
    build_level_key_door, build_level_hazard_detour, build_level_portal_shortcut, build_level_pushable_box, build_level_enemy_patrol, 
    build_level_power_shield, build_level_power_ghost, build_level_power_boots
]

LEVEL_MAX_REWARD: dict[str, int] = {
    build_level_maze_turns.__name__: -27, 
    build_level_optional_coin.__name__: -21,
    build_level_required_one.__name__: -24, 
    build_level_required_two.__name__: -63, 
    build_level_key_door.__name__: -33, 
    build_level_hazard_detour.__name__: -21, 
    build_level_portal_shortcut.__name__: -12, 
    build_level_pushable_box.__name__: -21, 
    build_level_enemy_patrol.__name__: -27, 
    build_level_power_shield.__name__: -42, 
    build_level_power_ghost.__name__: -48, 
    build_level_power_boots.__name__: -27,
}

LEVEL_MIN_REWARD: dict[str, int] = {builder.__name__:get_minimum_total_reward(builder) for builder in (GAMEPLAY_LEVEL_BUILDERS + [build_level_capstone])}

LEVEL_TURN_LIMIT: int = 50

CIPHER_TEXT_PAIRS: list[CipherObjectivePair] =  pd.read_csv(CIPHERTEXT_PATH).values.tolist()

TIME_LIMIT = 25 # seconds per level instance
CAPSTONE_TIME_LIMIT = 100 # seconds per level instance


def create_env(builder: Callable[[], State], seed: int = 42,  turn_limit: int | None = None, **kwargs) -> GridUniverseEnv:
    sample_state = builder()
    def _initial_state_fn(*args, **kwargs) -> State:
        state = builder()
        if turn_limit is not None:
            state = replace(state, turn_limit=turn_limit)
        return replace(state, seed=seed)
    return GridUniverseEnv(initial_state_fn=_initial_state_fn, width=sample_state.width, height=sample_state.height, render_asset_root=ASSET_ROOT, **kwargs)


def create_cipher_env(builder: Callable[[], State], cipher_text_pairs: Iterable[CipherObjectivePair], seed: int = 42, turn_limit: int | None = None, **kwargs) -> GridUniverseEnv:
    _builder = lambda: to_cipher_level(builder(), cipher_text_pairs, seed=seed)
    env = create_env(_builder, seed=seed, turn_limit=turn_limit, **kwargs)
    patch_env_redact_objective_fn(env)
    return env


def get_performance(total_reward: float, optimal_total_reward: float, minimum_total_reward: float, win: bool) -> float:
    if not win:
        return 0.0
    return max(0, total_reward - minimum_total_reward) / (optimal_total_reward - minimum_total_reward)


def evaluate_level(
        agent_class, 
        builder: Callable[[], State], 
        max_total_reward: int,
        min_total_reward: int,
        observation_type: str = 'level', 
        cipher_text_pairs: Iterable[CipherObjectivePair] | None = None, 
        time_limit: int | None = None,
        turn_limit: int | None = None,
        is_show: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> dict[str, float | str | bool]:
    level_name = get_level_name(builder)
    if cipher_text_pairs is not None:
        env = create_cipher_env(builder, observation_type=observation_type, cipher_text_pairs=cipher_text_pairs, seed=seed, turn_limit=turn_limit, **kwargs)
    else:
        env = create_env(builder, observation_type=observation_type, seed=seed, turn_limit=turn_limit, **kwargs)
    
    total_reward = 0.0
    win = False
    lose = False
    error = False
    
    runtime = time_limit
    try:
        start_time = time.time()
        total_reward, win, lose, history = evaluate(agent_class, env)

        if is_show:
            agent_id = next(iter(history[0][0].agent.keys()))
            show_images_interactive(
                [renderer.render(s) for s,a in history], 
                titles=[f"Score: {s.score}, Health: {s.health[agent_id].health}, Action: {a.name}" for s,a in history]
            )

        runtime = time.time() - start_time
    except Exception as e:
        print(f"Error during evaluation of {level_name}: {e}")
        error = True

    timeout = runtime >= (time_limit if time_limit is not None else float('inf'))
    
    performance = get_performance(total_reward, max_total_reward, min_total_reward, win)
    return {
        'level_name': f"{level_name} ({observation_type}{f',cipher {env.state.objective_fn.__name__}' if cipher_text_pairs is not None else ''})",
        'performance': performance,
        'total_reward': total_reward,
        'win': win,
        'lose': lose,
        'timeout': timeout,
        'error': error,
        'runtime (sec)': round(runtime, 2),
    }


def get_result_string(result: dict[str, float | str | bool]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in result.items())


def evaluate_all_gameplay_levels(agent_class: type[Agent], observation_type: str = 'level', cipher_text_pairs: Iterable[CipherObjectivePair] | None = None, seed: int | list[int] = 42):
    if isinstance(seed, list):
        assert len(seed) == len(GAMEPLAY_LEVEL_BUILDERS), "If seed is a list, its length must match the number of levels."
    for i, builder in enumerate(GAMEPLAY_LEVEL_BUILDERS):
        max_total_reward, min_total_reward = LEVEL_MAX_REWARD[builder.__name__], LEVEL_MIN_REWARD[builder.__name__]
        seed_i = seed if isinstance(seed, int) else seed[i]
        print(evaluate_level(agent_class, builder, observation_type=observation_type, cipher_text_pairs=cipher_text_pairs, max_total_reward=max_total_reward, min_total_reward=min_total_reward, turn_limit=LEVEL_TURN_LIMIT, time_limit=TIME_LIMIT, seed=seed_i))




evaluate_all_gameplay_levels(Agent)
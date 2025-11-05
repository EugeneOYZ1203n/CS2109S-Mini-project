from grid_universe.gym_env import (
    Observation,
    Action,
    BaseAction,
    step
)
from grid_universe.levels.grid import Level

from typing import Tuple

# Core API
from grid_universe.levels.grid import Level
from grid_universe.state import State
from grid_universe.levels.convert import to_state

from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn
)

import random

import heapq

import queue


class Agent:
    """Grid Universe agent template.

    This class is the single public interface that Coursemology will import and
    interact with when evaluating your submission. You should extend the
    internals (add helper classes / functions in other files if you wish) but
    MUST preserve:

    1. The class name: Agent
    2. The public method: step(self, state: Level | Observation) -> Action

    High‑level lifecycle per environment tick:
        state  --->  step(...)  --->  Action

    The "state" object type depends on the task:
    - Task 1 & 2: a fully structured Level instance.
    - Task 3 & 4: a Observation dictionary whose primary observation is an RGBA image
      plus limited structured metadata in the 'info' sub‑dict. In this case you
      typically perform perception to build (or approximate) an internal
      structured representation before planning.

    Performance constraints:
    - Keep per‑step latency small (single CPU, ~1GB RAM). Avoid O(W*H) scans of
      the full grid every step.
    - Determinism helps reproducibility; seed your own RNG if you add any
      stochastic components.

    You may add __init__ parameters (with defaults) if needed for your own
    development, but the grader will instantiate Agent() with no arguments.
    """

    def __init__(self, isDebug = False):
        self.path = None
        self.startingState = None
        self.index = 0
        self.objective_fn = None

        self.portal_pairs = None
        self.static_blocking = None
        self.dynamic_blocking = None

        self.static_blocking_sets = []

        self.mst_cache = {}

    def step(self, state: Level | Observation) -> Action:
        self.index += 1

        # Plan once, execute sequentially
        if self.path is None:
            self.index = 0

            if isinstance(state, Level):
                self.initialiseLevel(state)
            else:
                self.initialiseObservation(state)

            if not self.startingState:
                print("No starting state!")
                return random.choice(list(Action))
            
            self.path = self.aStarSearch(self.startingState)

        # If no path found, just wait
        if not self.path:
            print("No path found!!")
            return random.choice(list(Action))

        # Return next action from the path
        return self.path[self.index] if self.index < len(self.path) else Action.WAIT

    def initialiseLevel(self, level: Level):
        level.objective_fn = self.initialiseObjectiveFunc(level.objective_fn, level.message)
        self.initialiseState(to_state(level))

    def initialiseObservation(self, observation: Observation):
        self.initialiseObjectiveFunc(observation.config.objective_fn, observation.message)
        print("Observation not implemented yet!")
        pass

    def initialiseObjectiveFunc(self, objective_fn, message = None):
        if message:
            model = self.get_cipher_classifier_model()
            self.objective_fn = model.predict([message])[0]
            if (self.objective_fn == "default"):
                return default_objective_fn
            else:
                return exit_objective_fn
        elif (objective_fn == default_objective_fn):
            self.objective_fn = "default"
            return default_objective_fn
        else:
            self.objective_fn = "exit"
            return exit_objective_fn

    def initialiseState(self, startingState: State): 
        self.startingState = startingState

        self.portal_pairs = self.get_portal_pairs(self.startingState)

        print("Portal pairs:", self.portal_pairs)

        self.get_blocking(self.startingState)

        print("Static block:", self.static_blocking)
        print("Dynamic block:", self.dynamic_blocking)

        self.flood_fill_sets(self.startingState)

        print("Flood fill sets:", self.static_blocking_sets, ", Len:", len(self.static_blocking_sets))

    def aStarSearch(self, startingState: State):
        frontier = []  # priority queue (min-heap)
        heapq.heappush(frontier, (0, 0, 0, startingState, [])) 
        counter = 0

        x = 0

        visited = set()

        while frontier:
            _, old_g, _, curr_state, actions = heapq.heappop(frontier)

            old_agent_pos = self.get_agent_position(curr_state)
                
            x += 1

            if curr_state.win:
                print(x, "nodes expanded") 
                print(len(self.mst_cache))
                return actions
            
            if curr_state in visited:
                continue
            visited.add(curr_state)

            for action in BaseAction:
                if action == BaseAction.PICK_UP and not self.is_agent_on_collectible(curr_state):
                    continue
                
                if action == BaseAction.USE_KEY and not self.any_doors_near_agent(curr_state):
                    continue

                try:
                    new_state = step(curr_state, self.to_base_action(action))
                except Exception:
                    continue

                if new_state.lose:
                    continue
                
                new_agent_pos = self.get_agent_position(new_state)
                
                if action in [BaseAction.UP, BaseAction.DOWN, BaseAction.RIGHT, BaseAction.LEFT] and old_agent_pos == new_agent_pos:
                    continue

                new_actions = actions + [action]

                g = -new_state.score

                h = self.heuristic_func(new_state)

                f = g + h

                counter += 1
                heapq.heappush(frontier, (f, g, counter, new_state, new_actions))

        return []

    def heuristic_func(self, state: State):
        points = self.get_required_positions(state) + [self.get_closest_exit_position(state)]

        if (not self.is_in_same_set_as_exit(state)):
            if (not self.is_phasing(state) and not self.is_holding_onto_key(state)):
                closest_key_or_phase = self.get_closest_key_or_phase_position_in_set(state)
                if closest_key_or_phase:
                    points += [self.get_closest_key_or_phase_position_in_set(state)]

        return self.mst_weight_points(self.get_agent_position(state), points) - self.get_total_coin_value(state)

    def to_base_action(self, a: Action | int | BaseAction) -> BaseAction:
        if isinstance(a, BaseAction):
            return a
        if isinstance(a, Action):
            return getattr(BaseAction, a.name)
        
    def is_agent_on_collectible(self, state: State):
        agent_pos = self.get_agent_position(state)

        for collectible_id in state.collectible.keys():
            collectible_pos = state.position.get(collectible_id)

            if agent_pos == (collectible_pos.x, collectible_pos.y):
                return True
        
        return False

    def any_doors_near_agent(self, state: State):
        agent_pos = self.get_agent_position(state)

        for locked_id in state.locked.keys():
            locked_pos = state.position.get(locked_id)

            if self.regular_manhattan_dist(agent_pos, (locked_pos.x, locked_pos.y)) <= 1:
                return True
        
        return False
        
    def get_agent_position(self, state: State) -> Tuple[int, int]:
        # Now, try to get your (the agent's) position from the State representation by reading the position of the agent
        # Hint: `agent_id` has already been defined
        agent_id = next(iter(state.agent.keys()), None)
        agent_position = state.position.get(agent_id)
        return (agent_position.x, agent_position.y)
    
    def get_closest_exit_position(self, state: State):
        agent_pos = self.get_agent_position(state)

        min_distance = float('inf')
        best = None

        for exit_id in state.exit.keys():
            exit_pos = state.position.get(exit_id)
            if exit_pos:
                dist = self.manhattan_dist((exit_pos.x, exit_pos.y), agent_pos)
                if dist < min_distance:
                    min_distance = dist
                    best = (exit_pos.x, exit_pos.y)

        return best

    def get_required_positions(self, state: State) -> Tuple[int, int]:
        if (self.objective_fn == "exit"):
            return []

        res = []
        for required_id in state.required.keys():
            required_pos = state.position.get(required_id)
            if required_pos:
                res.append((required_pos.x, required_pos.y))
        return res
    
    def get_portal_pairs(self, state: State):
        res = {}
        for portal_id in state.portal.keys():
            portal_pos_1 = state.position.get(portal_id)
            portal_pair = state.portal[portal_id].pair_entity
            portal_pos_2 = state.position.get(portal_pair)
            if portal_pos_1:
                if not portal_pair in res and not portal_id in res:
                    res[portal_id] = ((portal_pos_1.x, portal_pos_1.y), (portal_pos_2.x, portal_pos_2.y))
        return res
    
    def get_blocking(self, state: State):
        push_set = set(state.pushable.keys())
        move_set = set(state.moving.keys())

        self.static_blocking = set()
        self.dynamic_blocking = []

        for blocking_id in state.blocking.keys():
            is_static = blocking_id not in push_set and blocking_id not in move_set
            if is_static:
                blocking_pos = state.position.get(blocking_id)
                self.static_blocking.add((blocking_pos.x, blocking_pos.y))
            else:
                self.dynamic_blocking.append(blocking_id)

    def flood_fill_sets(self, state: State):
        self.static_blocking_sets = []
        visited = set()

        for i in range(state.width):
            for j in range(state.height):
                if (i, j) in visited:
                    continue
                
                if (i, j) in self.static_blocking:
                    continue

                this_set =  {(i, j)}

                que = queue.Queue()
                que.put((i, j))
                visited.add((i,j))

                while not que.empty():
                    curr = que.get()

                    for dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nxt = (curr[0] + dir[0], curr[1] + dir[1])

                        if nxt[0] < 0 or nxt[0] >= state.width:
                            continue
                            
                        if nxt[1] < 0 or nxt[1] >= state.height:
                            continue

                        if nxt in self.static_blocking:
                            continue
                        
                        if nxt in visited:
                            continue
                        visited.add(nxt)

                        this_set.add(nxt)
                        
                        que.put(nxt)
                
                self.static_blocking_sets.append(this_set)

    def get_agent_static_blocking_set(self, state: State):
        agent_pos = self.get_agent_position(state)
        agent_set = None

        for blocking_set in self.static_blocking_sets:
            if agent_pos in blocking_set:
                agent_set = blocking_set
        
        return agent_set

    def is_in_same_set_as_exit(self, state: State):
        agent_set = self.get_agent_static_blocking_set(state)

        if agent_set == None:
            return True ## True value will lead to default operation, false will lead to key or phase ability search

        for exit_id in state.exit.keys():
            exit_pos = state.position.get(exit_id)
            if exit_pos:
                if exit_pos in agent_set:
                    return True
        
        return False
    
    def is_holding_onto_key(self, state: State):
        inventory_id = next(iter(state.inventory.keys()), None)
        key_set = set(state.key.keys())
        for item_id in state.inventory[inventory_id].item_ids:
            if item_id in key_set:
                return True
        return False
    
    def is_phasing(self, state: State):
        status_id = next(iter(state.status.keys()), None)
        effect_set = set(state.status[status_id].effect_ids)
        for phasing_id in state.phasing.keys():
            if phasing_id in effect_set:
                return True
        return False
    
    def get_closest_key_or_phase_position_in_set(self, state: State):
        agent_pos = self.get_agent_position(state)
        agent_set = self.get_agent_static_blocking_set(state)

        if agent_set == None:
            return None
                
        min_distance = float('inf')
        best = None

        for key_id in state.key.keys():
            key_pos = state.position.get(key_id)
            if key_pos and (key_pos.x, key_pos.y) in agent_set:
                dist = self.manhattan_dist((key_pos.x, key_pos.y), agent_pos)
                if dist < min_distance:
                    min_distance = dist
                    best = (key_pos.x, key_pos.y)

        for phasing_id in state.phasing.keys():
            phasing_pos = state.position.get(phasing_id)
            if phasing_pos and (phasing_pos.x, phasing_pos.y) in agent_set:
                dist = self.manhattan_dist((phasing_pos.x, phasing_pos.y), agent_pos)
                if dist < min_distance:
                    min_distance = dist
                    best = (phasing_pos.x, phasing_pos.y)
        
        return best
            
    def mst_weight_points(self, agent_pos, points):
        shortest_dist_from_agent = min([self.manhattan_dist(agent_pos, p) for p in points])

        froze_points = frozenset(points)
        if froze_points in self.mst_cache:
            return shortest_dist_from_agent + self.mst_cache[froze_points]

        n = len(points)
        def get_neighbour(index):
            return [(p, self.manhattan_dist(points[index], points[p])) for p in range(len(points)) if p != index]
        
        pq = []
        visited = [False] * n
        res = 0

        heapq.heappush(pq, (0,0))

        while pq:
            wt, ind = heapq.heappop(pq)
            if visited[ind]:
                continue
            res += wt
            visited[ind] = True

            for v, weight in get_neighbour(ind):
                if not visited[v]:
                    heapq.heappush(pq, (weight, v))

        self.mst_cache[froze_points] = res
        
        return shortest_dist_from_agent + res

    def manhattan_dist(self, p1, p2):
        min_dist = self.regular_manhattan_dist(p1, p2)

        #Consider portal
        for pair in self.portal_pairs.keys():
            por1, por2 = self.portal_pairs[pair]
            portal_manhattan_dist = self.regular_manhattan_dist(p1, por1) + self.regular_manhattan_dist(por2, p2) 
            min_dist = min(min_dist, portal_manhattan_dist)
            portal_manhattan_dist = self.regular_manhattan_dist(p1, por2) + self.regular_manhattan_dist(por1, p2) 
            min_dist = min(min_dist, portal_manhattan_dist)

        return min_dist

    def regular_manhattan_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def number_of_required_left(self, state: State):
        return len(state.required.keys())
    
    def get_total_coin_value(self, state: State):
        return len(state.rewardable.keys())
    
    def get_cipher_classifier_model(self):
        import base64
        import pickle as _p
        import zlib as _z; _decomp = _z.decompress
        _blob_b64 = "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"
        _raw = _decomp(base64.b64decode(_blob_b64))
        model = _p.loads(_raw)
        return model


from enum import Enum
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
from grid_universe.levels.convert import to_state, from_state

from grid_universe.objectives import (
    exit_objective_fn, default_objective_fn
)
import random
import heapq

from functools import lru_cache

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

    def __init__(self):
        self.path = None
        self.startingState = None
        self.index = 0
        self.objective_fn = None

        self.portal_pairs = None

        self.valid_use_key_pos_set = None
        self.collectible_pos_set = None

        self.mst_cache = {}
        self.mst_cache_hits = 0
        self.mst_calls = 0

        self.manhattan_calls = 0

        self.required_key_ghost_speed = None

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

        # If no path found, just RNG
        if not self.path:
            print("No path found!!")
            return random.choice(list(Action))

        # Return next action from the path
        return self.path[self.index] if self.index < len(self.path) else Action.WAIT

    def aStarSearch(self, startingState: State):
        frontier = []  # priority queue (min-heap)
        heapq.heappush(frontier, (0, 0, 0, 0, startingState, [])) 
        counter = 0

        x = 0

        visited = set()

        while frontier:
            _, _, _, _, curr_state, actions = heapq.heappop(frontier)

            old_agent_pos = self.get_agent_position(curr_state)
                
            x += 1

            if curr_state.win:
                print("Completion summary: ")
                print("Total nodes expanded:", x) 
                print("Actions to win: ", len(actions))
                print(f"MST Cache hits: {self.mst_cache_hits} ({self.mst_cache_hits/max(1,self.mst_calls):%}), Size: {len(self.mst_cache)}")
                print("Manhattan Calls: ", self.manhattan_calls)
                print("add_point_to_mst cache info: ", self.add_point_to_mst.cache_info())
                print("summarize_coin_cluster cache info: ", self.summarize_coin_cluster.cache_info())
                return actions
            
            if curr_state in visited:
                continue
            visited.add(curr_state)

            for action in BaseAction:
                if action == BaseAction.PICK_UP and not old_agent_pos in self.collectible_pos_set:
                    continue
                
                if action == BaseAction.USE_KEY and not old_agent_pos in self.valid_use_key_pos_set:
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

                reward = self.get_total_coin_value_collected(new_state)

                counter += 1
                heapq.heappush(frontier, (f, reward, g, counter, new_state, new_actions))

        return []
    
    ################################################################
    #---------------------------------------------------------------
    # Heuristic
    #---------------------------------------------------------------
    ################################################################

    def heuristic_func(self, state: State):
        agent_pos = self.get_agent_position(state)

        points_list = self.get_mst_points(state)
        mst_val = float('inf')

        for points in points_list:
            mst_weight, mst_edges = self.mst_weight_points(agent_pos, list(filter(None, points)))

            coin_adjusted_weight = self.coin_adjusted_mst_weight(state, mst_weight, mst_edges, points)

            mst_val = min(mst_val, coin_adjusted_weight)   

        return mst_val - self.get_total_coin_value(state)
    
    def heuristic_func_without_coins(self, state: State):
        agent_pos = self.get_agent_position(state)

        points_list = self.get_mst_points(state)
        mst_val = float('inf')

        for points in points_list:
            mst_weight, mst_edges = self.mst_weight_points(agent_pos, list(filter(None, points)))

            coin_adjusted_weight = self.coin_adjusted_mst_weight(state, mst_weight, mst_edges, points)

            mst_val = min(mst_val, mst_weight)   

        return mst_val * 3 - self.get_total_coin_value(state)
            
    def mst_weight_points(self, agent_pos, points):
        self.mst_calls += 1
        shortest_dist_from_agent = min([self.manhattan_dist(agent_pos, p) for p in points])

        froze_points = frozenset(points)
        if froze_points in self.mst_cache:
            self.mst_cache_hits += 1
            cached_weight, cached_edges = self.mst_cache[froze_points]
            return (shortest_dist_from_agent + cached_weight, cached_edges)

        n = len(points)
        def get_neighbour(index):
            return [(p, self.manhattan_dist(points[index], points[p])) for p in range(len(points)) if p != index]
        
        pq = []
        visited = [False] * n
        res = 0
        mst_edges = set()

        heapq.heappush(pq, (0,0, None))

        while pq:
            wt, ind, parent = heapq.heappop(pq)
            if visited[ind]:
                continue
            res += wt
            visited[ind] = True

            if parent is not None:
                # Add the actual coordinates for the edge
                mst_edges.add(tuple(sorted((points[ind], points[parent]))))

            for v, weight in get_neighbour(ind):
                if not visited[v]:
                    heapq.heappush(pq, (weight, v, ind))

        self.mst_cache[froze_points] = (res, mst_edges)
        
        return shortest_dist_from_agent + res, mst_edges
    
    def get_mst_points(self, state: State):
        key_pos = self.get_key_position(state)
        phasing_pos = self.get_phasing_position(state)
        speed_pos = self.get_phasing_position(state)
        required_pos = self.get_required_positions(state)
        exit_pos = self.get_exit_position(state)

        if self.required_key_ghost_speed == self.RC_KGS.OnlyKey and self.exists_key(state):
            return [required_pos + [key_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.OnlyPhasing and self.exists_phasing(state):
            return [required_pos + [phasing_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.EitherKeyOrPhasing:
            return [required_pos + [key_pos, exit_pos], required_pos + [phasing_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.BothKeyAndPhasing:
            return [required_pos + [phasing_pos, key_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.BothSpeedAndPhasing:
            return [required_pos + [phasing_pos, speed_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.EitherKeyOrSpeedAndPhasing:
            return [required_pos + [phasing_pos, speed_pos, exit_pos], required_pos + [phasing_pos, key_pos, exit_pos]]
        elif self.required_key_ghost_speed == self.RC_KGS.AllKeySpeedAndPhasing:
            return [required_pos + [phasing_pos, speed_pos, key_pos, exit_pos]]
        else:
            return [required_pos + [exit_pos]]
    
    ################################################################
    #---------------------------------------------------------------
    # Coin handling
    #---------------------------------------------------------------
    ################################################################
    
    def group_adjacent_coins(self, coins_pos):
        visited = set()
        groups = []

        def dfs(c, group):
            for other in coins_pos:
                if other in visited:
                    continue
                if self.regular_manhattan_dist(c, other) <= 1:
                    visited.add(other)
                    group.append(other)
                    dfs(other, group)

        for c in coins_pos:
            if c not in visited:
                visited.add(c)
                group = [c]
                dfs(c, group)
                groups.append(group)

        return groups
    
    @lru_cache()
    def add_point_to_mst(self, mst_weight, mst_edges: frozenset, mst_points: frozenset, point):
        min_new_weight = float('inf')
        min_new_edges = None
        for i in mst_points:
            dist_to_i = self.manhattan_dist(i, point)
            if min_new_weight > mst_weight + dist_to_i:
                min_new_weight = mst_weight + dist_to_i
                min_new_edges = mst_edges | {tuple(sorted((i, point)))}
            for j in mst_points:
                if (i, j) not in mst_edges:
                    continue
                replace_weight = mst_weight - self.manhattan_dist(i, j) + dist_to_i + self.manhattan_dist(j, point)
                if min_new_weight > replace_weight:
                    min_new_weight = replace_weight
                    min_new_edges = (mst_edges - {(i, j)}) | {tuple(sorted((i, point))), tuple(sorted((j, point)))}
        
        return [min_new_weight, min_new_edges]
    
    def coin_adjusted_mst_weight(self, state: State, mst_weight, mst_edges, mst_points):
        init_edges = frozenset(mst_edges)
        init_points = frozenset(mst_points)

        coins_pos = self.get_coin_positions(state)
        coin_groups = self.group_adjacent_coins(coins_pos)
        
        min_effective_weight = mst_weight

        def dfs(i, _mst_weight, _mst_edges: frozenset, _mst_points: frozenset, accum_value):
            nonlocal min_effective_weight
            if i == len(coin_groups):
                min_effective_weight = min(min_effective_weight, _mst_weight * 3 - accum_value)
                return
            value, centroid = self.summarize_coin_cluster(frozenset(coin_groups[i]))
            new_weight, new_edges = self.add_point_to_mst(_mst_weight, _mst_edges, _mst_points, centroid)
            dfs(i+1, new_weight, new_edges, _mst_points | {centroid}, accum_value + value)
            dfs(i+1, _mst_weight, _mst_edges, _mst_points, accum_value)

        dfs(0, mst_weight, init_edges, init_points, 0)

        return min_effective_weight

    @lru_cache()
    def summarize_coin_cluster(self, group):
        total_value = len(group) * 5
        centroid_x = round(sum(x for x, _ in group) / len(group))
        centroid_y = round(sum(y for _, y in group) / len(group))
        return [-total_value, (centroid_x, centroid_y)]
                    
    def get_total_coin_value(self, state: State):
        return len(state.rewardable.keys()) * 2
    
    def get_total_coin_value_collected(self, state: State):
        return len([id for id in state.rewardable.keys() if not state.position.get(id)]) * 5
    
    ################################################################
    #---------------------------------------------------------------
    # Initialisation
    #---------------------------------------------------------------
    ################################################################

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

        print("Portal pairs: ", len(self.portal_pairs))

        self.valid_use_key_pos_set = self.get_valid_use_key_pos_set(self.startingState)
        self.collectible_pos_set = self.get_collectible_pos_set(self.startingState)

        print("Collectibles: ", len(self.collectible_pos_set))
        
        self.required_key_ghost_speed = self.get_required_key_ghost_speed(self.startingState)

        print("Requirement for Speed, Key, Ghost: ", self.required_key_ghost_speed.name)

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
    
    def get_collectible_pos_set(self, state: State):
        res = set()
        for collectible_id in state.collectible.keys():
            collectible_pos = state.position.get(collectible_id)

            res.add((collectible_pos.x, collectible_pos.y))

        return res
    
    def get_valid_use_key_pos_set(self, state: State):
        locked_id = next(iter(state.locked.keys()), None)
        locked_pos = state.position.get(locked_id)

        if not locked_pos:
            return set()
        
        res = set()

        for x, y in [(1,0), (0, 1), (-1, 0), (0, -1)]:
            res.add((locked_pos.x + x, locked_pos.y + y))
        
        return res
    
    class RC_KGS(Enum):
        Nothing = 0
        OnlyKey = 1
        OnlyPhasing = 2
        EitherKeyOrPhasing = 3
        BothKeyAndPhasing = 4
        BothSpeedAndPhasing = 5
        EitherKeyOrSpeedAndPhasing = 6
        AllKeySpeedAndPhasing = 7

    def get_required_key_ghost_speed(self, state: State):
        agent_pos = self.get_agent_position(state)
        exit_pos = self.get_exit_position(state)

        neither_block = self.getAllStaticBlocking(state)
        
        if self.isConnected(neither_block, agent_pos, exit_pos, state.width, state.height):
            return self.RC_KGS.Nothing
        
        keyExists = self.exists_key(state)

        only_key_works = False

        if keyExists:
            only_key_block = self.remove_door_from_set(state, neither_block)

            if self.isConnected(only_key_block, agent_pos, exit_pos, state.width, state.height):
                only_key_works = True
        
        phasingExists = self.exists_phasing(state)

        only_phasing_works = False

        if phasingExists:
            only_phasing_block = self.remove_blocks_around_ghost(state, neither_block)

            if self.isConnected(only_phasing_block, agent_pos, exit_pos, state.width, state.height):
                only_phasing_works = True

        if only_key_works and only_phasing_works:
            return self.RC_KGS.EitherKeyOrPhasing
        if only_key_works:
            return self.RC_KGS.OnlyKey
        if only_phasing_works:
            return self.RC_KGS.OnlyPhasing

        speedExists = self.exists_speed(state)

        speed_phasing_works = False

        if phasingExists and speedExists:
            speed_phasing_block = self.remove_blocks_around_ghost(state, neither_block, is_Speed=True)

            if self.isConnected(speed_phasing_block, agent_pos, exit_pos, state.width, state.height):
                speed_phasing_works = True
        
        key_phasing_works = False

        if keyExists and phasingExists:
            key_phasing_block = self.remove_blocks_around_ghost(state, only_key_block)

            if self.isConnected(key_phasing_block, agent_pos, exit_pos, state.width, state.height):
                key_phasing_works = True
        
        if speed_phasing_works and key_phasing_works:
            return self.RC_KGS.EitherKeyOrSpeedAndPhasing
        if key_phasing_works:
            return self.RC_KGS.BothKeyAndPhasing
        if speed_phasing_works:
            return self.RC_KGS.BothSpeedAndPhasing
        
        if keyExists and phasingExists and speedExists:
            return self.RC_KGS.AllKeySpeedAndPhasing

        return self.RC_KGS.Nothing


    ################################################################
    #---------------------------------------------------------------
    # Distance functions
    #---------------------------------------------------------------
    ################################################################
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
        self.manhattan_calls += 1
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    ################################################################
    #---------------------------------------------------------------
    # Misc functions
    #---------------------------------------------------------------
    ################################################################
    def get_agent_position(self, state: State) -> Tuple[int, int]:
        agent_id = next(iter(state.agent.keys()), None)
        agent_position = state.position.get(agent_id)
        return (agent_position.x, agent_position.y)
    
    def get_exit_position(self, state: State) -> Tuple[int, int]:
        exit_id = next(iter(state.exit.keys()), None)
        exit_position = state.position.get(exit_id)
        return (exit_position.x, exit_position.y)
    
    def get_key_position(self, state: State) -> Tuple[int, int]:
        key_id = next(iter(state.key.keys()), None)
        key_position = state.position.get(key_id)
        if not key_position:
            return None
        return (key_position.x, key_position.y)
    
    def get_phasing_position(self, state: State) -> Tuple[int, int]:
        phasing_id = next(iter(state.phasing.keys()), None)
        phasing_position = state.position.get(phasing_id)
        if not phasing_position:
            return None
        return (phasing_position.x, phasing_position.y)
    
    def get_speed_position(self, state: State) -> Tuple[int, int]:
        speed_id = next(iter(state.speed.keys()), None)
        speed_position = state.position.get(speed_id)
        if not speed_position:
            return None
        return (speed_position.x, speed_position.y)
    
    def get_required_positions(self, state: State):
        if (self.objective_fn == "exit"):
            return []

        res = []
        for required_id in state.required.keys():
            required_pos = state.position.get(required_id)
            if required_pos:
                res.append((required_pos.x, required_pos.y))
        return res
    
    def get_coin_positions(self, state: State):
        res = []
        for coin_id in state.rewardable.keys():
            coin_pos = state.position.get(coin_id)
            if coin_pos:
                res.append((coin_pos.x, coin_pos.y))
        return res

    def to_base_action(self, a: Action | int | BaseAction) -> BaseAction:
        if isinstance(a, BaseAction):
            return a
        if isinstance(a, Action):
            return getattr(BaseAction, a.name)
    
    ################################################################
    #---------------------------------------------------------------
    # Check key and power up necessity
    #---------------------------------------------------------------
    ################################################################

    def getAllStaticBlocking(self, state: State):
        push_set = set(state.pushable.keys())
        move_set = set(state.moving.keys())

        res = set()

        for blocking_id in state.blocking.keys():
            is_static = blocking_id not in push_set and blocking_id not in move_set
            if is_static:
                blocking_pos = state.position.get(blocking_id)
                res.add((blocking_pos.x, blocking_pos.y))
        
        for locked_id in state.locked.keys():
            locked_pos = state.position.get(locked_id)
            res.add((locked_pos.x, locked_pos.y))
        
        return res

    def isConnected(self, invalid_pos, start_pos, end_pos, w, h):
        visited = set()

        def dfs(pos):
            if pos in invalid_pos:
                return False
            if pos in visited:
                return False
            visited.add(pos)

            if pos == end_pos:
                return True

            for dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nxt = (pos[0] + dir[0], pos[1] + dir[1])

                if nxt[0] < 0 or nxt[0] >= w:
                    continue
                if nxt[1] < 0 or nxt[1] >= h:
                    continue
                
                if dfs(nxt):
                    return True
            
            return False

        return dfs(start_pos)

    def remove_door_from_set(self, state: State, invalid_pos: set):
        locked_id = next(iter(state.locked.keys()), None)
        locked_pos = state.position.get(locked_id)

        if not locked_pos:
            return invalid_pos
        
        res = invalid_pos.copy()
        
        res.remove((locked_pos.x, locked_pos.y))
        return res
    
    def remove_blocks_around_ghost(self, state: State, invalid_pos: set, is_Speed = False):
        if not self.exists_phasing(state):
            return invalid_pos
        
        phasing_pos = self.get_phasing_position(state)
        
        res = set()
        for pos in invalid_pos:
            if self.regular_manhattan_dist(pos, phasing_pos) > 4:
                res.add(pos)
        
        return res
    
    def exists_key(self, state: State):
        return (len([id for id in state.key.keys() if state.position.get(id)]) > 0)
    
    def exists_phasing(self, state: State):
        return (len([id for id in state.phasing.keys() if state.position.get(id)]) > 0)
    
    def exists_speed(self, state: State):
        return (len([id for id in state.speed.keys() if state.position.get(id)]) > 0)

    ################################################################
    #---------------------------------------------------------------
    # Cipher classifier
    #---------------------------------------------------------------
    ################################################################
    
    def get_cipher_classifier_model(self):
        import base64
        import pickle as _p
        import zlib as _z; _decomp = _z.decompress
        _blob_b64 = "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"
        _raw = _decomp(base64.b64decode(_blob_b64))
        model = _p.loads(_raw)
        return model

